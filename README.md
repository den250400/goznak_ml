# MEL-spectrograms classification and denoising

![Demo](https://github.com/den250400/goznak_ml/blob/main/denoising_demo.png "Demo")

В этом репозитории представлены модели для классификации MEL-спектрограмм на зашумленный/незашумленные классы и для очищения MEL-спектрограмм от шумов (denoising). В качестве архитектуры используется LSTM, который обрабатывает последовательность векторов (80, 1) - распределение интенсивности звука по частотам (мелам) в данный момент времени. В задаче классификации, последнее значение hidden state линейно преобразуется в единственное число - score по зашумленности/незашумленности. В denoising, каждое значение hidden state линейно преобразуется в вектор (80, 1) - распределение интенсивности звука по частотам (мелам), очищенное от шумов.


Кроме этого, в репозитории представлены inference-скрипты для работы с аудио, позволяющие оценить качество работы модели не только на спектрограммах, но и на аудиофайлах.


__Модель классификации обучалась на 50 эпохах, Accuracy = 97.5%__

__Модель denoising обучалась на 10 эпохах, MSE = 0.039__

## Установка всех необходимых библиотек
```
pip3 install -r requirements.txt
```

## Обучение и тестирование
Данные для обучения и валидации должны находиться в папке data в корне проекта, а сама папка должна иметь следующую структуру:

```
- train\
- - noisy\
- - - speaker1_id\
- - - ...
- - clean\
- - - speaker1_id\
- - - ...
- val\
- - noisy\
- - - speaker1_id\
- - - ...
- - clean\
- - - speaker1_id\
- - - ...
```

После обучения state dict модели сохраняется в папку models, и может быть в дальнейшем загружен другими скриптами для тестирования и inference. В этом репозитории в папке models уже лежат две предобученные модели: ```classifier.pth``` и ```denoiser.pth```.

---

__Обучение модели классификации__
```
python3 train_classification.py --epochs=50 --dataset_path='./data' --model_filename='classifier.pth'
```
Аргументы командной строки:

```epochs``` - количество эпох обучения

```dataset_path``` - путь к папке с обучающим (train) и валидационным (val) датасетом

```model_filename``` - название файла с весами (state dict) модели в папке models

__Обучение denoising-модели__
```
python3 train_denoising.py --epochs=10 --dataset_path='./data' --model_filename='denoiser.pth'
```

---

__Тестирование модели классификации, выводит на экран accuracy на выбранном датасете__
```
python3 eval_classification.py --dataset_path='./data/val' --model_filename='classifier.pth'
```
Аргументы командной строки

```dataset_path``` - путь к папке с валидационным/тестовым датасетом

```model_filename``` - название файла с весами (state dict) модели в папке models

__Тестирование denoising-модели, выводит на экран средний MSE на выбранном датасете__
```
python3 eval_denoising.py --dataset_path='./data/val' --model_filename='denoiser.pth'
```
После вычисления среднего MSE, скрипт будет выводить на экран три спектрограммы для каждого элемента датасета: зашумленная, очищенная от шумов при помощи нейросети, и эталонная, чистая спектрограмма
## Inference
Первый inference-скрипт (```inference_spectrogram.py```) принимает на вход .npy-файл с зашумленной спектрограммой, очищает его от шумов при помощи обученной модели, сохраняет очищенную спектрограмму на диск, и выводит на экран 2 спектрограммы: входную и выходную
```
python3 inference_spectrogram.py --input_path='./data/noisy.npy' --model_filename='denoiser.pth' --output_path='./data/predicted.npy'
```

---

Второй inference-скрипт (```inference_audio.py```) демонстрирует работу denoising-модели на реальном аудиофайле: скрипт принимает на вход зашумленное аудио, и сохраняет в папку ./data его denoised-версию predicted.wav. Преобразование спектрограммы в аудиофайл осуществляется при помощи алгоритма Гриффина-Лима.
```
python3 inference_audio.py --input_path='./data/noisy.wav' --model_filename='denoiser.pth' --output_path='./data/predicted.wav'
```
Аргументы командной строки

```input_path``` - путь к исходному, зашумлённому аудиофайлу

```output_path``` - путь для сохранения очищенного от шумов аудиофайла

---

Третий inference-скрипт (```inference_from_dataset.py```) берет случайную спектрограмму из выбранного датасета, прогоняет ее через denoising-модель, и сохраняет на диск 3 wav-файла:

* clean.wav - аудиофайл, соответствующий "чистой" спектрограмме (преобразование в аудио происходит с помощью алгоритма Гриффина-Лима)

* noisy.wav - аудиофайл, соответствующий зашумленной спектрограмме

* predicted.wav - аудиофайл, соответствующий denoised-версии спектрограммы

```
python3 inference_from_dataset.py --dataset_path='./data/val' --model_filename='denoiser.pth' --clean_path='./data/clean.wav' --noisy_path='./data/noisy.wav' --predicted_path='./data/predicted.wav'
```
