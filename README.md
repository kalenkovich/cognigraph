# cognigraph
Inverse-modelling-related capabilities of cognigraph

## Инструкции по установке
1. **Питон и пакеты.** 
Самый простой вариант - через среду conda. 
Наиболее универсальным вариантом оказалось не использование .yml, 
а последовательная установка с помощью conda/pip. Список команд находится в
[create_conda_environment.txt](scripts/create_conda_environment.txt).

2. **Репозиторий.** Часть зависимостей организована через подмодули git. Для
того, чтобы они загрузились вместе с текущим репозиторием при клонировании 
необходимо добавить флаг `--recursive`. 

3. **Необходимые файлы.** Программа использует файлы из датасета _sample_, 
распространяемого с пакетом _mne-python_. Чтобы не качать все файлы (датасет
лежит на osf.io, загрузка с которого крайне неторопливая), можно скачать
урезанную версию 
[отсюда](https://drive.google.com/open?id=1D0jI_Z5EycI8JwJbYOAYdSycNGoarmP-). 
Папку _MNE-sample-data_ из архива надо скопировать в то же место, куды бы ее 
загрузил _mne-python_. Чтобы узнать, что это за место, не скачивая датасет, 
нужно сделать следующее: 

    ```
    from mne.datasets import sample
    print(sample.data_path(download=False, verbose=False))
    ```
    Папку _MNE-sample-data_ из архива копируем в выведенный путь.

## Пробный запуск

1. Скачиваем тестовую запись 
[отсюда](https://drive.google.com/drive/folders/1y0oLYXyzqAZJLCAokmAt4OjBDxVtIN93?usp=sharing).

2. Запускаем _iPython_ в корне репозитория c флагом `--gui=qt`:
    ```
    ipython --gui=qt
    ```
    
3. Пишем в переменную `launch_test_filepath` пусть к любому из файлов в папке
с тестовой записью:
    ```
    launch_test_filepath = r'<path_to_a_file>'
    ```
    
4. Прогоняем тестовый скрипт:
    ```
    from scripts.launch_test import *
    ```
    
5. В результате должно появиться окно с корой и настройками. Запускаем обработку:
    ```
    timer.start()
    ```
    На коре начнет изменяться активность.




