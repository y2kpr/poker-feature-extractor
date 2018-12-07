import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

confLSTM = [[2665,   78,  206,  162,  146],
             [ 143, 5302,  799,   56,  221],
             [ 233,  454, 5412,  228,  229],
             [ 277,  89,  269, 2549,  121],
             [ 126,   53,  351,   26, 9346]]

confGRU = [[2860,   96,  133,  103,   65],
           [ 106, 6174,  136,   33,   72],
           [ 134,  223, 6067,   40,   92],
           [ 313,   91,  144, 2645,  112],
           [  31,   38,   43,  26, 9764]]

 # [[2860   96  133  103   65]
 # [ 106 6174  136   33   72]
 # [ 134  223 6067   40   92]
 # [ 313   91  144 2645  112]
 # [  31   38   43   26 9764]]

df_cm = pd.DataFrame(confLSTM, index = [i for i in ['call', 'check', 'bet', 'raise', 'fold']],
                  columns = [i for i in ['call', 'check', 'bet', 'raise', 'fold']])

# plt.matshow(confLSTM, cmap='binary', interpolation='None')
# plt.show()
# fig, ax = plt.subplots()
plt.figure(figsize = (8,6))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
# ax.get_yaxis().set_major_formatter(
#     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# plt.ticklabel_format(style='plain', axis='y')
# ax.ticklabel_format(useOffset=False)
plt.show()
