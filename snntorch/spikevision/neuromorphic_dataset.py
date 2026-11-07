import os
import torch
import torch.utils.data as data
from ._transforms import Compose
from tqdm import tqdm
import tarfile
import zipfile
import gzip
import hashlib
import ntpath
import stat
import urllib.error
import urllib.parse
import urllib.request


class _SecureRedirect(urllib.request.HTTPRedirectHandler):
    """Reject HTTPS downgrades before following a redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if (urllib.parse.urlsplit(req.full_url).scheme.lower() == "https"
                and urllib.parse.urlsplit(newurl).scheme.lower() != "https"):
            raise urllib.error.URLError("Refusing an insecure HTTPS redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _contained_path(root, name):
    """Reject portable absolute/traversal paths and existing escaping symlinks."""
    name = os.fspath(name)
    parts = name.replace("\\", "/").split("/")
    if ntpath.splitdrive(name)[0] or name.startswith(("/", "\\")) or ".." in parts:
        raise ValueError("Unsafe archive or download path: " + name)
    base = os.path.realpath(root)
    target = os.path.realpath(os.path.join(base, name))
    if os.path.commonpath([base, target]) != base:
        raise ValueError("Path escapes destination: " + name)
    return target


def _extract_tar(archive, destination):
    # A documented standard-library data filter is required, including on Python
    # versions with security backports. Older runtimes fail closed.
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Safe TAR extraction requires tarfile.data_filter")
    for member in archive.getmembers():
        _contained_path(destination, member.name)
        if not (member.isfile() or member.isdir()):
            raise ValueError("Archive links and special files are not supported")
        tarfile.data_filter(member, destination)
    archive.extractall(path=destination, filter="data")


# Adapted from https://github.com/nmi-lab/torchneuromorphic by Emre Neftci and Clemens Schaefer

DEFAULT_ROOT = "data/"


def download_url(url, root, filename=None, md5=None, total_size=None):

    """Download a file from a url and place it in root.

    :param url: URL to download file from
    :type url: string

    :param root: Directory to place downloaded file in
    :type root: string

    :param filename: Name to save the file under. If None, use the basename of the URL
    :type filename: string, optional

    :param md5: MD5 checksum of the download. If None, do not check
    :type md5: string, optional
    """

    import shutil
    import tempfile
    import urllib.parse
    import urllib.request

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(urllib.parse.urlsplit(url).path)
    fpath = _contained_path(root, filename)
    os.makedirs(root, exist_ok=True)
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # The HTTPS opener retains Python's default certificate validation and checks
    # every redirect before connecting. Never retry a secure URL over HTTP.
    opener = urllib.request.build_opener(_SecureRedirect())
    fd, partial = tempfile.mkstemp(prefix=".snntorch-", dir=root)
    try:
        with os.fdopen(fd, "wb") as output, opener.open(url, timeout=30) as response:
            shutil.copyfileobj(response, output)
        if not check_integrity(partial, md5):
            raise RuntimeError("File not found or corrupted.")
        os.replace(partial, fpath)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def download_and_extract_archive(
    url,
    download_root,
    extract_root=None,
    filename=None,
    md5=None,
    remove_finished=False,
):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(urllib.parse.urlsplit(url).path)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    _extract_archive(archive, extract_root, remove_finished)


# backwards compatibility
def identity(x):
    return x


class NeuromorphicDataset(data.Dataset):
    _repr_indent = 4

    def __init__(
        self,
        root=None,
        transforms=None,
        transform=None,
        target_transform=None,
        transform_train=None,
        transform_test=None,
        target_transform_train=None,
        target_transform_test=None,
    ):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        if root is not None:
            if not os.path.isfile(root):
                if self.download_and_create:
                    self._download()  # check if raw files exist in root
                    self._create_hdf5()  #
                else:
                    raise Exception(
                        "File {} does not exist and download_and_create is False".format(
                            root
                        )
                    )

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can "
                "be passed as argument"
            )

        # for backwards-compatibility
        if transform is None:
            transform = identity
        if target_transform is None:
            target_transform = identity

        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def extra_repr(self):
        return ""

    def _check_exists(self):
        res_ = [os.path.exists(d) for d in self.resources_local]
        res = all(res_)
        if res is False:
            print("The following files do not exist, will attempt download:")
            for i, r in enumerate(res_):
                if not r:
                    print(self.resources_local[i])
        return res

    def _extract_exists(self):
        res_ = [os.path.exists(d) for d in self.resources_local_extracted]
        res = all(res_)
        if res is False:
            for _, _, filename in self._resources_url:
                extract_root = self.directory
                archive = os.path.join(extract_root, filename).replace("\\", "/")
                print("Extracting {} to {}...".format(archive, extract_root))
                _extract_archive(archive, extract_root, remove_finished=False)
        return res

    def _download(self):
        if self._check_exists():
            if self._extract_exists():
                return True
        else:
            os.makedirs(self.directory, exist_ok=True)
            for url, md5, filename in self._resources_url:
                download_and_extract_archive(
                    url, download_root=self.directory, filename=filename, md5=md5
                )
            return False

    def _create_hdf5(self):
        raise NotImplementedError()

    def target_transform_append(self, transform):
        if transform is None:
            return
        if self.target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = Compose([self.target_transform, transform])

    def transform_append(self, transform):
        if transform is None:
            return
        if self.transform is None:
            self.transform = transform
        else:
            self.transform = Compose([self.transform, transform])


def _extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, "r") as tar:
            _extract_tar(tar, to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, "r:gz") as tar:
            _extract_tar(tar, to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, "r:xz") as tar:
            _extract_tar(tar, to_path)
    elif _is_gzip(from_path):
        to_path = _contained_path(
            to_path, os.path.splitext(os.path.basename(from_path))[0]
        )
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, "r") as z:
            for member in z.infolist():
                _contained_path(to_path, member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Archive links are not supported")
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(
                self.target_transform, "Target transform: "
            )

        return "\n".join(body)


def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")


def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def check_integrity(fpath, md5=None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update
