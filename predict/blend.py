import glob
import pandas as pd
from omegaconf import OmegaConf
import os

conf = OmegaConf.from_cli()
print(OmegaConf.to_yaml(conf))

target_csv = glob.glob(os.path.join(conf.csv_dir, '*.csv'))

target_df = []
for f in target_csv:
    target_df.append(pd.read_csv(f))

sub_df = target_df[0].copy()
sub_df['target'] = 0.0

for idx, df in enumerate(target_df):
    sub_df['target'] += df['target']/len(target_df)

sub_df.to_csv(conf.sub_name, index=False)
