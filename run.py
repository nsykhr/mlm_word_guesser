import argparse
from modules import GameHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', '-p', type=str, default='rubert_cased_L-12_H-768_A-12_pt')
    parser.add_argument('--max_rounds', '-r', type=int, default=10)
    parser.add_argument('--prob_threshold', '-t', type=float, default=0.5)
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('-n', type=int, default=5)
    args = parser.parse_args()

    game_handler = GameHandler(args.bert_path, args.max_rounds, args.prob_threshold, args.device)
    game_handler.play_game(args.n)


if __name__ == '__main__':
    main()
