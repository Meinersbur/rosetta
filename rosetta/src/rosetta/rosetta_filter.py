def add_filter_args(parser):
    parser.add_argument('--filter-program-include', '--includeprogram', action='append', default=[],
                        help="Only look into filenames that contain this substring")
    parser.add_argument('--filter-program-exclude', '--excludeprogram', action='append', default=[],
                        help="Only exclude filenames that contain this substring")
    parser.add_argument('--filter-ppm-include', '--includeppm', action='append', default=[],
                        help="Only look into programming models that contain this substring")
    parser.add_argument('--filter-ppm-exclude', '--excludeppm', action='append', default=[],
                        help="Only exclude programming models that contain this substring")


def match_program_filter(b, args):
    filter_program_include = args.filter_program_include
    filter_program_exclude = args.filter_program_exclude
    basename = b.basename
    if filter_program_include:
        if not any(f in basename for f in filter_program_include):
            return False
        return True
    if filter_program_exclude:
        if any(f in basename for f in filter_program_exclude):
            return False
        return True
    return True


def match_ppm_filter(b, args):
    filter_ppm_include = args.filter_ppm_include
    filter_ppm_exclude = args.filter_ppm_exclude
    ppm = b.ppm
    if filter_ppm_include:
        if not any(f in ppm for f in filter_ppm_include):
            return False
        return True
    if filter_ppm_exclude:
        if any(f in ppm for f in filter_ppm_exclude):
            return False
        return True
    return True


def match_filter(b, args):
    return match_program_filter(b, args) and match_ppm_filter(b, args)
