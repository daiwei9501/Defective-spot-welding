import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report

x = np.load('x_val.npy')
y_true = np.load('y_val.npy')

#model = load_model('models/cnnbest.h5')
#model = load_model('models/resnet50_best.h5')
#model = load_model('models/transefer_resnet50.h5')
model = load_model('models/transfer_resnet50_GAN_best.h5')

y_pred = np.argmax(model.predict(x), axis=1)
print(classification_report(y_true, y_pred, digits =3))

print('acc:', accuracy_score(y_true,y_pred))
print('recall', recall_score(y_true,y_pred,average='macro'))
print('preci', precision_score(y_true, y_pred,average='macro'))
print('f1:', f1_score(y_true, y_pred,average='macro'))

import matplotlib.pyplot as plt
import matplotlib as mpl
#classes = ['edge weld', 'double', 'pseudo welds', 'burr', 'distortion', 'burn through','good']
confusion = confusion_matrix(y_true, y_pred)
# 绘制热度图
plt.imshow(confusion, cmap='Blues')
plt.xticks([])
plt.yticks([])
plt.colorbar()
#plt.xlabel('True label')
#plt.ylabel('Predict label')

# 显示数据
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index],ha='center')

# 显示图片
plt.show()