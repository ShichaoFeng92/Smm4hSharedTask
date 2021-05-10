import numpy as np
f=open('cos_matrix_task2_en.csv')
f.readline()
remove_data=set()
for line_id,line in enumerate(f):
    x=line.strip().split(',')[1:]
    a=np.asarray(x,float)
    if a[line_id]==1:
        remove_data.add(line_id)
        position
        if np.count_nonzero(np.where(a<0.1))>10:
            remove_data.add()

     
