


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true") 
    args = parser.parse_args() 
    if args.local:
        print("Local Config")
    else:
        print("GPU Config")