/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2006  Alexey N. Salnikov (salnikov@cmc.msu.ru)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#ifndef __AUTO_
#define __AUTO_

#include <math.h>
#include <fftw3.h>

#include "my_time.h"

/// @brief Тип для функции, вычисляющей некий скалярный параметр, характеризующий окно заданной длины.
/// Параметры: window - окно со значениями задержки, для которого рассчитывается параметр.
///            window_length - длина данного окна.
typedef double (*ScalarAlgorithmCounter)(double* window,
                                         int window_length);

/// @brief Тип для функции, вычисляющей длину вектора мощностей гармоник, который получается для окна заданной длины.
/// Параметры: window_length - длина окна, для которого будет считаться спектр.
typedef int (*PowerSpectrumLengthCounter)(int window_length);

/// @brief Тип для функции, вычисляющей вектор мощностей гармоник некого спектрального
/// представления окна заданной длины.
/// Параметры: window - окно со значениями задержки, для которого рассчитывается параметр.
///            window_length - длина данного окна.
/// Возращаемое значение: вектор мощностей гармоник
typedef double* (*PowerSpectrumCounter)(double* window,
                                int window_length,
                                int power_spectrum_length);

/// @brief Тип для функции, вычисляющей некую норму вектора.
/// Параметры: vector - вектор, для которого считается норма.
///            vector_length - длина данного вектора.
typedef double (*NormCounter)(double* vector,
                              int vector_length);

/// @brief Тип для функции, вычисляющей некое расстояние между двумя векторами.
/// Параметры: first_vector, second_vector - вектора, расстояние между которыми необходимо найти.
///            vector_length - длина данных векторов.
///       p.s: Предполагается, что данные векторы одинаковой длины.
typedef double (*DistanceCounter)(double* first_vector,
                                  double* second_vector,
                                  int vector_length);


/// @brief Структура для параметров алгоритма, ответственных за выбор окон.
struct WindowSelectionParameters
{
  /// @brief Отношение длины выбираемых окон к текущему количеству итераций (0 < cur_iter < 1).
  double ratio;

  /// @brief Количество выбираемых окон.
  int window_amount;
};

/// @brief Структура для функций, вычисляющие основные величины для "спектрального" алгоритма.
struct SpectrumAlgorithmCounters
{
  /// @brief Функция для подсчета длины векторов спектра.
  PowerSpectrumLengthCounter spectrum_length_counter;

  /// @brief Функция, вычисляющая вектор мощностей гармоник.
  PowerSpectrumCounter spectrum_counter;

  /// @brief Функция для подсчета нормы векторов спектра. 
  NormCounter norm_counter;  

  /// @brief Функция для подсчета расстояния между векторами спектра.
  DistanceCounter distance_counter;
};

/// @brief Перечислимый тип для общих видов доступных алгоритмов.
enum AlgorithmGeneralType
{
  /// @brief Алгоритм автоматизации определения количества измерений задержки не используется.
  kNoAlgo,

  /// @brief Алгоритм, основанный на сравнении скалярных величин в окнах.
  kScalarAlgo,

  /// @brief Алгоритм, основанный на сравнении векторов мощностей гармоник в спектральном представлении окон.
  kSpectrumAlgo,

};

/// @brief Перечислимый тип для доступных алгоритмов.
enum AlgorithmType
{
  /// @brief Алгоритм автоматизации определения количества измерений задержки не используется.
  kEmptyAlgo,

  /// @brief Алгоритм, основанный на сравнении скалярных величин в окнах.
  /// В данном случае скалярной величиной является миинимальное значение в окне.
  kScalarMinAlgo,

  /// @brief Алгоритм, основанный на сравнении скалярных величин в окнах.
  /// В данном случае скалярной величиной является значение математического ожидания в окне.
  kScalarAvgAlgo,

  /// @brief Алгоритм, основанный на сравнении скалярных величин в окнах.
  /// В данном случае скалярной величиной является значение дисперсии в окне.
  kScalarDevAlgo,

  /// @brief Алгоритм, основанный на сравнении скалярных величин в окнах.
  /// В данном случае скалярной величиной является значение медианы в окне.
  kScalarMedAlgo,

  /// @brief Алгоритм, основанный на сравнении векторов мощностей гармоник в спектральном представлении окон.
  /// В данном случае спектрально разложение делается с помощью Быстрого Преобразования Фурье (FFT).
  /// В качестве норм и расстояний для векторов мощностей гармоник выступают обычные евклидовы норма и расстояние.
  kSpectrumFFTAlgo,

  /// @brief Алгоритм, основанный на сравнении векторов мощностей гармоник в спектральном представлении окон.
  /// В данном случае спектрально разложение делается с помощью Быстрого Преобразования Фурье (FFT).
  /// В качестве норм и расстояний для векторов мощностей гармоник выступают евклидовые норма и расстояние
  /// между векторами спектра, из которых предварительно исключили первый элемент (самую медленную гармонику).
  kModifiedSpectrumFFTAlgo
};

/// @brief Структура, содержащая всю необходимую информацию для функционирования
/// алгоритмов автоматического определения количества измеренйи задержки.
struct AlgorithmMainInfo
{
  /// @brief Частота проверки критерия останова.
  /// (Проверка осуществляется раз в frequency измерений задержки)
  int frequency;

  /// @brief Общий вид используемого алгоритма.
  enum AlgorithmGeneralType algorithm_general_type;

  /// @brief Параметры выбора окон.
  struct WindowSelectionParameters window_selection_parameters;

  /// @brief Функция подсчета скалярного параметра
  /// (Используется если algorithm_general_type = kScalarAlgo)
  ScalarAlgorithmCounter scalar_algorithm_counter;

  /// @brief Структура для функций, вычисляющие основные величины для "спектрального" алгоритма.
  /// (Используется если algorithm_general_type = kSpectrumAlgo)
  struct SpectrumAlgorithmCounters spectrum_algorithm_counters;

  /// @brief Целевое значение критерия останова.
  double target_criterion_value;
};


/// Функции, вычисляющие скалярные параметры, характеризующие окна.
/// Эти функции соответствуют типу ScalarAlgorithmCounter.

/// @brief Функция, вычисляющая минимальное значение данного окна. 
double min_counter(double* window,
                   int window_length);

/// @brief Функция, вычисляющая математическое ожидание данного окна.
double average_counter(double* window,
                       int window_length);

/// @brief Функция, вычисляющая дисперсию данного окна.
double deviation_counter(double* window,
                         int window_length);

/// @brief Функция, вычисляющая медиану данного окна.
double median_counter(double* window,
                      int window_length);

/// Функции, вычисляющие длину вектора мощностей гармоник для некоторого спектрального разложения окна заданной длины.
/// Эти функции соответствуют типу SpectrumLengthCounter. Каждой такой функции соответствует парная функция типа SpectrumCounter.

/// @brief Функция, вычисляющая длину вектора мощностей гармоник, получающегося при применении
/// Быстрого Преобразования Фурье (FFT) к окну с заданной длинной. 
int fft_power_spectrum_length_counter(int window_length);


/// Функции, вычисляющие вектор мощностей гармоник для некоторого спектрального разложения данного окна.
/// Эти функции соответствуют типу SpectrumCounter.

/// @brief Функция, вычисляющая вектор мощностей гармоник для спектрального разложения,
/// сделанного с помощью Быстрого Преобразования Фурье (FFT). 
double* fft_power_spectrum_counter(double* window,
                                   int window_length,
                                   int power_spectrum_length);


/// @brief Функции, вычисляющие некую норму данного вектора.
/// Эти функции соответствуют типу NormCounter.

/// @brief Функция, вычисляющая евклидову норму данного вектора. 
double euclidean_norm_counter(double* vector,
                              int vector_length);

/// @brief Функция, вычисляющая евклидову норму вектора, полученного из данного
/// в результате исключения первого элемента вектора. 
double modified_euclidean_norm_counter(double* vector,
                              int vector_length);


/// @brief Функции, вычисляющие некое расстояние между данными векторами.
/// Эти функции соответствуют типу DistanceCounter.

/// @brief Функция, вычисляющая евклидово расстояние между данными векторами. 
double euclidean_distance_counter(double* first_vector,
                                  double* second_vector,
                                  int vector_length);

/// @brief Функция, вычисляющая евклидово расстояние между векторами, полученными
/// из данных в результате исключения первых элементов данных векторов. 
double modified_euclidean_distance_counter(double* first_vector,
                                  double* second_vector,
                                  int vector_length);


/// @brief Функция для подсчета критерия, основанного на сравнении значений скалярных величин
/// @param cur_iter Текущее количество измерений задержки (итераций).
/// @param window_selection_parameters Параметры выбора окон.
/// @param scalar_parameter_counter Указатель на функцию для подсчета скалярного параметра, характеризующего окно. 
/// @param tmp_results Вектор с текущими результатми измерения задержки (имеет длину cur_iter).
/// @return Значение критерия, основанного на сравнении значений скалярных величин.
double scalar_criterion(int cur_iter,
                        struct WindowSelectionParameters window_selection_parameters,
                        ScalarAlgorithmCounter scalar_parameter_counter,
                        px_my_time_type *tmp_results);

/// @brief Функция для подсчёта критерия, основанного на сравнении
/// векторов мощностей гармоник спектральных представлений окон.
/// Суть критерия: чем меньше отношение среднего расстояния между векторами мощностей гармоник к
/// средней норме векторов мощностей гармоник, тем ближе окна друг к другу. 
/// @param cur_iter Текущее количество измерений задержки (итераций).
/// @param window_selection_parameters Параметры выбора окон.
/// @param counters Структура с указателями на функции для подсчёта основных величин "спектрального" алгоритма.
/// @param tmp_results Вектор с текущими результатми измерения задержки (имеет длину cur_iter).
/// @return Значения критерия, основанного на сравнении векторов мощностей гармоник спектральных представлений окон.
double spectrum_criterion(int cur_iter,
                        struct WindowSelectionParameters window_selection_parameters,
                        struct SpectrumAlgorithmCounters counters,
                        px_my_time_type *tmp_results);


/// @brief Функция, заполняющая структуру AlgorithmMainInfo по входным данным.
/// @param algorithm_type Тип используемого алгоритма.
/// @param frequency Частота проверки условия останова (проверка осуществляется раз в frequency измерений задержки).
/// @param window_amount Количество окон. 
/// @param ratio Отношение длины выбираемых окон к текущему количеству итераций (0 < cur_iter < 1).
/// @param target_criterion_value Целевое значение критерия останова.
/// @return Указатель на заполненную структуру AlgorithmMainInfo.
struct AlgorithmMainInfo* get_algorithm_main_info(enum AlgorithmType algorithm_type,
                                                  int frequency,
                                                  int window_amount,
                                                  double ratio,
                                                  double target_criterion_value);

#endif /* __AUTO_ */