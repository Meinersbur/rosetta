import re


def add_filter_args(parser):
    parser.add_argument(
        '--filter-include-program-substr',
        action='append',
        default=[],
        help="Only look into filenames that contain this substring",
    )
    parser.add_argument(
        '--filter-include-program-exact',
        action='append',
        default=[],
        help="Only look into filename that exactly matches this string",
    )
    parser.add_argument(
        '--filter-include-program-regex',
        action='append',
        default=[],
        help="Only look into filenames that match this regular expression",
    )
    parser.add_argument(
        '--filter-exclude-program-substr',
        action='append',
        default=[],
        help="Only exclude filenames that contain this substring",
    )
    parser.add_argument(
        '--filter-exclude-program-exact',
        action='append',
        default=[],
        help="Only exclude filename that exactly matches this string",
    )
    parser.add_argument(
        '--filter-exclude-program-regex',
        action='append',
        default=[],
        help="Only exclude filenames that match this regular expression",
    )
    parser.add_argument(
        '--filter-include-ppm-substr',
        action='append',
        default=[],
        help="Only look into programming models that contain this substring",
    )
    parser.add_argument(
        '--filter-include-ppm-exact',
        action='append',
        default=[],
        help="Only look into programming model that exactly matches this string",
    )
    parser.add_argument(
        '--filter-include-ppm-regex',
        action='append',
        default=[],
        help="Only look into programming models that match this regular expression",
    )
    parser.add_argument(
        '--filter-exclude-ppm-substr',
        action='append',
        default=[],
        help="Only exclude programming models that contain this substring",
    )
    parser.add_argument(
        '--filter-exclude-ppm-exact',
        action='append',
        default=[],
        help="Only exclude programming model that exactly matches this string",
    )
    parser.add_argument(
        '--filter-exclude-ppm-regex',
        action='append',
        default=[],
        help="Only exclude programming models that match this regular expression",
    )


def apply_program_filter(benchmarks, args):
    include_program_substr = args.filter_include_program_substr
    include_program_exact = args.filter_include_program_exact
    include_program_regex = args.filter_include_program_regex
    exclude_program_substr = args.filter_exclude_program_substr
    exclude_program_exact = args.filter_exclude_program_exact
    exclude_program_regex = args.filter_exclude_program_regex

    for f in exclude_program_substr:
        benchmarks = [b for b in benchmarks if not f in b.basename]
    for f in exclude_program_exact:
        benchmarks = [b for b in benchmarks if not b.basename == f]
    try:
        for f in exclude_program_regex:
            benchmarks = [b for b in benchmarks if not re.match(f, b.basename)]
    except re.error:
        print(f"Regex error in --filter-exclude-program-regex: {str(re.error)}")
    if not (include_program_substr or include_program_exact or include_program_regex):
        return benchmarks
    filtered_benchmarks = []
    for i, b in enumerate(benchmarks):
        already_included = False
        for f in include_program_substr:
            if f in b.basename:
                already_included = True
                filtered_benchmarks.append(b)
        for f in include_program_exact:
            if b.basename == f and not already_included:
                already_included = True
                filtered_benchmarks.append(b)
        try:
            for f in include_program_regex:
                if re.match(f, b.basename) and not already_included:
                    already_included = True
                    filtered_benchmarks.append(b)
        except re.error:
            print(f"Regex error in --filter-include-program-regex: {str(re.error)}")
    return filtered_benchmarks


def apply_ppm_filter(benchmarks, args):
    include_ppm_substr = args.filter_include_ppm_substr
    include_ppm_exact = args.filter_include_ppm_exact
    include_ppm_regex = args.filter_include_ppm_regex
    exclude_ppm_substr = args.filter_exclude_ppm_substr
    exclude_ppm_exact = args.filter_exclude_ppm_exact
    exclude_ppm_regex = args.filter_exclude_ppm_regex

    for f in exclude_ppm_substr:
        benchmarks = [b for b in benchmarks if not f in b.ppm]
    for f in exclude_ppm_exact:
        benchmarks = [b for b in benchmarks if not b.ppm == f]
    try:
        for f in exclude_ppm_regex:
            benchmarks = [b for b in benchmarks if not re.match(f, b.ppm)]
    except re.error:
        print(f"Regex error in --filter-exclude-ppm-regex: {str(re.error)}")
    if not (include_ppm_substr or include_ppm_exact or include_ppm_regex):
        return benchmarks
    filtered_benchmarks = []
    for i, b in enumerate(benchmarks):
        already_included = False
        for f in include_ppm_substr:
            if f in b.ppm:
                already_included = True
                filtered_benchmarks.append(b)
        for f in include_ppm_exact:
            if b.ppm == f and not already_included:
                already_included = True
                filtered_benchmarks.append(b)
        try:
            for f in include_ppm_regex:
                if re.match(f, b.ppm) and not already_included:
                    already_included = True
                    filtered_benchmarks.append(b)
        except re.error:
            print(f"Regex error in --filter-include-ppm-regex: {str(re.error)}")
    return filtered_benchmarks


def get_filtered_benchmarks(benchmarks, args):
    program_benchmarks = apply_program_filter(benchmarks, args)
    return apply_ppm_filter(program_benchmarks, args)
