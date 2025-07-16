import argparse
import yaml




def main():
    parser = argparse.ArgumentParser(description="Parser for the config YAML file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    llm = config.get("llm", "Default Name")
    model = config.get("model", 0)
    input_path = config.get("input_path", "Unknown")
    output_path = config.get("output_path", "Unknown")
    

    print(f"{llm} {model} {input_path} {output_path}")


if __name__=="__main__":
    main()

