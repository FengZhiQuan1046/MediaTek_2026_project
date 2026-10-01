"""Show raw progress in the terminal and write readable events to a log file."""

from pathlib import Path
import re
import sys


ANSI = re.compile(rb"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS = re.compile(
    rb"(?:\b\d{1,3}%[^\n]*\||\[\d{2}:\d{2}[^\]\n]*/s\]|"
    rb"\b(?:Sanity Checking|Training|Validation|"
    rb"Testing|Predicting|Epoch \d+|LLaRA SASRec|Loading checkpoint shards)"
    rb"(?: DataLoader \d+)?:\s*\|)"
)


def clean_record(record: bytes) -> bytes:
    plain = ANSI.sub(b"", record)
    return b"" if PROGRESS.search(plain) or not plain.strip() else plain


def main() -> None:
    pending = bytearray()
    with Path(sys.argv[1]).open("wb") as log:
        while chunk := sys.stdin.buffer.read1(65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            for byte in chunk:
                if byte in (10, 13):
                    cleaned = clean_record(bytes(pending))
                    if cleaned:
                        log.write(cleaned + b"\n")
                        log.flush()
                    pending.clear()
                else:
                    pending.append(byte)
        cleaned = clean_record(bytes(pending))
        if cleaned:
            log.write(cleaned + b"\n")


if __name__ == "__main__":
    main()
