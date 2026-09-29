"""Download the pinned public source and verify the publisher's checksum."""
import hashlib
import urllib.request
from pathlib import Path

from deployguard.data.prepare_cicd import SOURCE_SHA256

URL = 'https://data.mendeley.com/public-files/datasets/mggwn7rj9f/files/19b6dd3d-d6cc-4965-b6f5-afda17583e18/file_downloaded'


def main():
    path = Path('data/raw/final_research_dataset_MASTER.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix('.download')
        urllib.request.urlretrieve(URL, temporary)
        with temporary.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != SOURCE_SHA256:
            raise ValueError('Downloaded source checksum mismatch')
        temporary.replace(path)
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != SOURCE_SHA256:
            raise ValueError('Existing source checksum mismatch; file left untouched')
    print(f'Verified {path}')


if __name__ == '__main__':
    main()
