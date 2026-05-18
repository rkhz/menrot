import os
import argparse

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import menrot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'], help='Data split')
    parser.add_argument('--task', required=True, choices=['cognitive', 'symbolic', 'renderer'], help='Task to build')
    args = parser.parse_args()
    args = parser.parse_args()

    wrkdir =  os.environ.get("WRKDIR")
    if wrkdir is None:
        wrkdir = input("WRKDIR is not set, enter a path to store the data: ")
    wrkdir = os.path.abspath(wrkdir)

    if args.task == 'cognitive':
        print('cognitive')
        builder = menrot.utils.data.MenrotCognitiveBuilder(
            root_dir=os.path.join(wrkdir, "data"),
            split=args.split
        )
    elif args.task == 'symbolic':
        print('symbolic')
        builder = menrot.utils.data.MenrotSymbolicBuilder(
            root_dir=os.path.join(wrkdir, "data"),
            split=args.split
        )
    elif args.task == 'renderer':
        print('renderer')
        builder = menrot.utils.data.MenrotRendererBuilder(
            root_dir=os.path.join(wrkdir, "data"),
            split= args.split,
            num_views=25
        )
        
    builder.run()