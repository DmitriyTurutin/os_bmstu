#include <iostream>
#include <omp.h> // подключаем open mp

int main() {
    #pragma omp parallel // чтобы выполнялось параллельно
    {
        int thread_num = omp_get_thread_num(); // записываем в переменную # текущего потока
        std::cout << "Привет мир! Я поток " << thread_num << std::endl; // выводим сообщения "Привет мир" и номер потока, который выполнил
    }
    return 0;
}
