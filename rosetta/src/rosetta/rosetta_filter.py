import re


def add_filter_args(parser):
    parser.add_argument('--include-program', action='append', default=[],
                        help="Only look into filenames that contain this substring")
    parser.add_argument('--include-program-exact', action='append', default=[],
                        help="Only look into filename that exactly matches this string")
    parser.add_argument('--include-program-regex', action='append', default=[],
                        help="Only look into filenames that match this regular expression")
    parser.add_argument('--exclude-program', action='append', default=[],
                        help="Only exclude filenames that contain this substring")
    parser.add_argument('--exclude-program-exact', action='append', default=[],
                        help="Only exclude filename that exactly matches this string")
    parser.add_argument('--exclude-program-regex', action='append', default=[],
                        help="Only exclude filenames that match this regular expression")
    parser.add_argument('--include-ppm', action='append', default=[],
                        help="Only look into programming models that contain this substring")
    parser.add_argument('--include-ppm-exact', action='append', default=[],
                        help="Only look into programming model that exactly matches this string")
    parser.add_argument('--include-ppm-regex', action='append', default=[],
                        help="Only look into programming models that match this regular expression")
    parser.add_argument('--exclude-ppm', action='append', default=[],
                        help="Only exclude programming models that contain this substring")
    parser.add_argument('--exclude-ppm-exact', action='append', default=[],
                        help="Only exclude programming model that exactly matches this string")
    parser.add_argument('--exclude-ppm-regex', action='append', default=[],
                        help="Only exclude programming models that match this regular expression")


def match_program_filter(b, args):
    include_program = args.include_program
    include_program_exact = args.include_program_exact
    include_program_regex = args.include_program_regex
    exclude_program = args.exclude_program
    exclude_program_exact = args.exclude_program_exact
    exclude_program_regex = args.exclude_program_regex
    basename = b.basename
    if include_program:
        if not any(f in basename for f in include_program):
            return False
        return True
    if include_program_exact:
        if not any(basename == f for f in include_program_exact):
            return False
        return True
    if include_program_regex:
        try:
            if not any(re.match(f, basename) for f in include_program_regex):
                return False
            return True
        except re.error:
            return False
    if exclude_program:
        if any(f in basename for f in exclude_program):
            return False
        return True
    if exclude_program_exact:
        if any(basename == f for f in exclude_program_exact):
            return False
        return True
    if exclude_program_regex:
        try:
            if any(re.match(f, basename) for f in exclude_program_regex):
                return False
            return True
        except re.error:
            return False
    return True


def match_ppm_filter(b, args):
    include_ppm = args.include_ppm
    include_ppm_exact = args.include_ppm_exact
    include_ppm_regex = args.include_ppm_regex
    exclude_ppm = args.exclude_ppm
    exclude_ppm_exact = args.exclude_ppm_exact
    exclude_ppm_regex = args.exclude_ppm_regex
    ppm = b.ppm
    if include_ppm:
        if not any(f in ppm for f in include_ppm):
            return False
        return True
    if include_ppm_exact:
        if not any(ppm == f for f in include_ppm_exact):
            return False
        return True
    if include_ppm_regex:
        try:
            if not any(re.match(f, ppm) for f in include_ppm_regex):
                return False
            return True
        except re.error:
            return False
    if exclude_ppm:
        if any(f in ppm for f in exclude_ppm):
            return False
        return True
    if exclude_ppm_exact:
        if any(ppm == f for f in exclude_ppm_exact):
            return False
        return True
    if exclude_ppm_regex:
        try:
            if any(re.match(f, ppm) for f in exclude_ppm_regex):
                return False
            return True
        except re.error:
            return False
    return True


def match_filter(b, args):
    return match_program_filter(b, args) and match_ppm_filter(b, args)
