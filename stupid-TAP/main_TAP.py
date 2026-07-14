from cli import build_parser
from tap_runner import run_tap


if __name__ == '__main__':
    args = build_parser().parse_args()
    run_tap(args)
