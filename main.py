"""
main.py — project entry point
------------------------------
Usage:
    python main.py train          # train from scratch
    python main.py eval           # evaluate val set (confusion matrix)
    python main.py predict <img>  # classify a single image
"""

import sys
from initialisation import NUM_CLASSES, CLASS_NAMES

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'

    if mode == 'train':
        from two_stage_training import train
        train()

    elif mode == 'eval':
        from SaveAndReload import load_model
        from SingleImageInference import evaluate_folder
        model = load_model('dress_classifier.pth')
        evaluate_folder('dataset', 'val', model)

    elif mode == 'predict':
        if len(sys.argv) < 3:
            print("Usage: python main.py predict <image_path>")
            sys.exit(1)
        from SaveAndReload import load_model
        from SingleImageInference import predict_style
        model  = load_model('dress_classifier.pth')
        result = predict_style(sys.argv[2], model)
        print(f"\nPrediction : {result['predicted_class']}")
        print(f"Confidence : {result['confidence']:.1%}")
        print("All scores :")
        for cls, score in result['all_scores'].items():
            bar = "█" * int(score * 30)
            print(f"  {cls:<14} {bar:<30}  {score:.1%}")

    else:
        print(f"Unknown mode '{mode}'. Use: train | eval | predict <img>")
        sys.exit(1)

if __name__ == '__main__':
    main()