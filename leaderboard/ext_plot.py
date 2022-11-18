from utils import *
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]=True

credential_file = './credentials.json'

def plot_html(args):
    dfs = read_dataframes_by_time_from_gcp(args.credentials)
    output_file(filename='leaderboard_plotting.html', title='PyHealth Leaderboard')

    bokeh_figures = []

    for task in args.tasks:
        df = get_typed_df_with_time(dfs, task)
        bokeh_figure = generate_bokeh_figure(df)
        bokeh_figures.append(bokeh_figure)

    save(column(bokeh_figures))


def construct_args():
    parser.add_argument("--credentials", type=str, default=credential_file)
    parser.add_argument("--tasks", type=list,
                        default=[
                            "drugrec",
                            "lenOfStay",
                            "mortality",
                            "readmission"
                        ])


def main():
    args = construct_args()
    plot_html(args)


if __name__ == '__main__':
    main()


    
