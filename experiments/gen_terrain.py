from src.experiment.terrain_generation import build_generation_parser, generate_one, resolve_output_layout


def main():
    ap = build_generation_parser(single_seed=True)
    args = ap.parse_args()
    H, W, size_tag, _ = resolve_output_layout(args)
    out_path = generate_one(args, args.seed)
    print('saved:', out_path, flush=True)
    print('map_size:', {'H': H, 'W': W, 'tag': size_tag}, flush=True)
    print('terrain_type:', args.terrain_type, flush=True)


if __name__ == '__main__':
    main()
