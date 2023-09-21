
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="average models")
    parser.add_argument("--mdl1", required=True, type=str)
    parser.add_argument("--mdl2", required=True, type=str)
    parser.add_argument("--avg_ratio", required=True, type=float)
    parser.add_argument("--out_mdl", required=True, type=str)

    args = parser.parse_args()

    print(".........Loading models........")
    model1_all_states = torch.load(args.mdl1, map_location=torch.device("cpu"))
    model1 = model1_all_states["model"]
    model2 = torch.load(args.mdl2, map_location=torch.device("cpu"))["model"]
    print("Loading Done!")

    print("Averaging Models:")
    assert args.avg_ratio >= 0 and args.avg_ratio <= 1, "Ration must be in [0,1]."
    avg = dict()
    for param_name in model1.keys():
        avg[param_name] = args.avg_ratio * model1[param_name] + (1 - args.avg_ratio) * model2[param_name]

    model1_all_states['model'] = avg
    torch.save(model1_all_states, args.out_mdl)
    print("Done!")


if __name__ == "__main__":
    main()
