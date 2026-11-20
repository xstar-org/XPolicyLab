"""Compare complete evaluator results; never use the agent's self-assessment."""
import argparse
import json
import math
from pathlib import Path


def outcomes(path):
    data = json.loads(path.read_text())
    details = data.get('details', {})
    if data.get('eval_time') != 50 or len(details) != 50:
        raise ValueError(f'{path}: expected 50 completed evaluator episodes')
    result = {}
    for row in details.values():
        layout = row.get('layout_id')
        if type(layout) is not int or layout in result or type(row.get('success')) is not bool:
            raise ValueError(f'{path}: missing/duplicate layout_id or nonboolean success')
        result[layout] = row['success']
    if set(result) != set(range(50)):
        raise ValueError(f'{path}: expected the frozen layout IDs 0..49')
    return result


def summary(values):
    n, k = len(values), sum(values)
    z, p = 1.959963984540054, k / n
    denominator = 1 + z*z/n
    middle = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return dict(successes=k, episodes=n, success_rate=p,
                wilson_95=[max(0., middle-half), min(1., middle+half)])


def compare(original, official, layout_ids=None):
    if set(original) != set(range(50)) or set(official) != set(range(50)):
        raise ValueError('Both complete 50-layout arms are required, including for subset reporting')
    ids = list(range(50)) if layout_ids is None else list(layout_ids)
    if (not ids or any(type(i) is not int or not 0 <= i < 50 for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Comparison layout IDs must be unique nonempty frozen IDs')
    a_only = sum(original[i] and not official[i] for i in ids)
    b_only = sum(official[i] and not original[i] for i in ids)
    discordant = a_only + b_only
    p_value = min(1., 2 * sum(math.comb(discordant, k)
                  for k in range(min(a_only, b_only)+1)) / 2**discordant)
    result = dict(original=summary([original[i] for i in ids]),
                official_only=summary([official[i] for i in ids]),
                difference_original_minus_official=sum(int(original[i])-int(official[i]) for i in ids)/len(ids),
                paired=dict(both_success=sum(original[i] and official[i] for i in ids),
                            original_only=a_only, official_only=b_only,
                            both_fail=sum(not original[i] and not official[i] for i in ids),
                            mcnemar_exact_two_sided_p=p_value))
    if layout_ids is not None:
        result['layout_ids'] = sorted(ids)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--official-only', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layout-ids', help='Optional comma-separated subset, after both full arms pass validation')
    args = parser.parse_args()
    ids = [int(value) for value in args.layout_ids.split(',')] if args.layout_ids else None
    result = compare(outcomes(args.original), outcomes(args.official_only), ids)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
