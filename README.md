# MEL-spectrograms classification and denoising
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

Обучение модели классификации
```
python3 train_classification.py --epochs=50 --dataset_path='./data' --model_filename='classifier.pth'
```
Обучение denoising-модели
```
python3 train_denoising.py --epochs=10 --dataset_path='./data' --model_filename='denoiser.pth'
```
Тестирование модели классификации, выводит на экран accuracy на выбранном датасете
```
python3 train_denoising.py --dataset_path='./data/val' --model_filename='classifier.pth'
```
Тестирование denoising-модели, выводит на экран средний MSE на выбранном датасете
```
python3 train_denoising.py --dataset_path='./data/val' --model_filename='denoiser.pth'
```
## Inference
Первый inference-скрипт (```inference_audio.py```) демонстрирует работу denoising-модели на реальном аудиофайле: скрипт принимает на вход зашумленное аудио, и сохраняет в папку ./data его denoised-версию predicted.wav. Преобразование спектрограммы в аудиофайл осуществляется при помощи алгоритма Гриффина-Лима.
```
python3 inference_audio.py --input_path='./data/noisy.wav' --model_filename='denoiser.pth' --output_path='./data/predicted.wav'
```
Второй inference-скрипт (```inference_dataset.py```) берет случайную спектрограмму из выбранного датасета, прогоняет ее через denoising-модель, и сохраняет на диск 3 wav-файла:

* clean.wav - аудиофайл, соответствующий "чистой" спектрограмме (преобразование в аудио происходит с помощью алгоритма Гриффина-Лима)

* noisy.wav - аудиофайл, соответствующий зашумленной спектрограмме

* predicted.wav - аудиофайл, соответствующий denoised-версии спектрограммы

```
python3 inference_dataset.py --dataset_path='./data/val' --model_filename='denoiser.pth' --clean_path='./data/clean.wav' --noisy_path='./data/noisy.wav' --predicted_path='./data/predicted.wav'
```
