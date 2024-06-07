import os
from tqdm import tqdm
# import date
import datetime


target = '/WAVE/workarea/users/sli13/AIC23_MTPC/stcra/results/submissions'
files = os.listdir(target)

output = '/WAVE/workarea/users/sli13/AIC23_MTPC/stcra/results/track1-%s.txt'

# get the date as string in terms of mm-dd
date_str = datetime.date.today().strftime('%m-%d')
output_str = output % date_str

with open(output_str, 'w+') as f:
    
    
    # simply append all files
    for file in tqdm(files):
        # print(file)
        with open(os.path.join(target, file), 'r') as f2:
            for line in f2:
                f.write(line)
                
