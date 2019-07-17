from glob import glob
import pandas as pd
from pdb import set_trace as trace
import os

FILES_PATH = '/specific/netapp5_2/gamir/achiya/Sandisk/new_data/PC3/'

all_files = glob(FILES_PATH + '*.csv')
files_count = 0
for file in all_files:
    dut_num = file.split('/')[-1].split('_')[1][3:]
    output_path = os.path.join(FILES_PATH, 'split', 'DUT' + dut_num)
    os.makedirs(output_path, exist_ok=True)
    cur_df = pd.read_csv(file, index_col=False)
    cur_df = cur_df[cur_df['Bank'] == 2]
    cur_df['file_idx'] = cur_df.apply(lambda row: int(row.BLK / 25), axis=1)
    output_files_count = 0
    files_groupby = cur_df.groupby('file_idx')
    for group_idx, out_file_group in files_groupby:
        with open(os.path.join(output_path, '{}.csv'.format(group_idx)), 'w') as out_f:
            out_f.write(out_file_group.drop(columns=['file_idx']).to_csv(index=False))
        output_files_count += 1
        if output_files_count % 100 == 0:
            print('finished {} out of 473 output files for input file no. {} out of {}'
                  .format(output_files_count, files_count, len(all_files)))

    files_count += 1