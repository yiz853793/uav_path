from src.experiment.terrain import (
    build_generation_parser,
    resolve_output_layout,
    generate_one,
)


def main():
    ap = build_generation_parser(single_seed=False)
    args = ap.parse_args()

    H, W, size_tag, out_subdir = resolve_output_layout(args)

    seeds = list(range(args.seed_from, args.seed_to))
    out_paths = []

    for i, seed in enumerate(seeds, 1):
        path = generate_one(args, int(seed))
        out_paths.append(path)
        print(f'[{i}/{len(seeds)}] saved: {path}', flush=True)

    print(f'Done. Generated {len(out_paths)} terrains into {out_subdir}', flush=True)
    print('map_size:', {'H': H, 'W': W, 'tag': size_tag}, flush=True)
    print('terrain_type:', args.terrain_type, flush=True)


if __name__ == '__main__':
    main()
