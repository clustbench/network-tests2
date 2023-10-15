#include <array>
#include <iostream>

template <int Column, int N>
constexpr const std::array<std::array<int, N-1>, N-1>& CreateSmallerMatrix(const std::array<std::array<int, N>, N>& matrix) {
    std::array<std::array<int, N-1>, N-1> new_matrix;
    int offset = 0;
    for(int i = 1; i < N; i++) {
        for(int j = 0; j < size-1; j++) {
            //Пропустить col-ый столбец
            if(j == col) {
                offset = 1; //Встретили нужный столбец, проускаем его смещением
            }

            new_matrix[i-1][j] = matrix[i][j + offset];
        }
    }
    return &new_matrix;
}

template <int Column, int N>
constexpr int MakeExpansion(const std::array<std::array<int, N>, N>& matrix){
    return Column == N ? 0 : (I % 2 == 0 ? matrix[0][Column]*Det<N-1>(CreateSmallerMatrix<0, Column, N>(matrix)) : -matrix[0][I]*Det<N-1>(CreateSmallerMatrix<0, Column, N>(matrix)));
}

template <int N>
constexpr int Det(const std::array<std::array<int, N>, N>& matrix) {
    return N == 1 ? matrix[0][0] : (N == 2 ? matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0] : MakeExpansion<0, N>(matrix));
}

template <int N>
constexpr int Trace(const std::array<std::array<int, N>, N>& matrix){
    return N == 1 ? matrix[0][0] : matrix[0][0]
}

int main() {
    constexpr std::array<std::array<int, 3>, 3> matrix = {{
        {0, 1, 2},
        {1, 2, 3},
        {2, 3, 7}
    }};
    constexpr int result = Det<3>(matrix);
    std::cout << result << std::endl;
    return 0;
}