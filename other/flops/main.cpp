#include <iostream>
#include <unistd.h>
#include <ctime>

uint64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec*1e9 + (uint64_t)start.tv_nsec;
}

int main()
{
    uint64_t start = nanos();
    sleep(1);
    uint64_t end = nanos();

    auto s = (end - start)*1e-9;
    std::cout << "Заняло: " << s << " сек.";
    return 0;
}