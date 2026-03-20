#include "delay_measurements_amount_auto.h"

/// @brief Функция, вычисляющая минимальное значение данного окна.
double min_counter(double* window,
                   int window_length)
{
  double min_value = window[0];

  for (int i = 1; i < window_length; i++)
  {
    if (window[i] < min_value)
    {
      min_value = window[i];
    }
  }

  return min_value;
}

/// @brief Функция, вычисляющая математическое ожидание данного окна.
double average_counter(double* window,
                       int window_length)
{
  double avg_value = 0.0;

  for (int i = 0; i < window_length; i++)
  {
    avg_value += window[i];
  }
  avg_value /= window_length;

  return avg_value;
}

/// @brief Функция, вычисляющая дисперсию данного окна.
double deviation_counter(double* window,
                         int window_length)
{
  double deviation_value = 0.0;
  double avg_squares_value = 0.0;
  double avg_value = 0.0;

  for (int i = 0; i < window_length; i++)
  {
    avg_value += window[i];
    avg_squares_value += window[i]*window[i];
  }
  deviation_value = avg_squares_value - avg_value*avg_value;

  return deviation_value;
}

/// @brief Компоратор для сортироваки массива для вычисления медианы.
int cmp_to_count_median(const void *a, const void *b)
{
    double val_a=*(double *)a;
    double val_b=*(double *)b;

    if((val_a - val_b)>0) return 1;
    else if((val_a - val_b)<0) return -1;
    else return 0;
}

/// @brief Функция, вычисляющая вектор мощностей гармоник для спектрального разложения,
/// сделанного с помощью Быстрого Преобразования Фурье (FFT). 
double median_counter(double* window,
                      int window_length)
{
  double *sorted_window = (double*)malloc(window_length*sizeof(double));

  for (int i = 0; i < window_length; i++)
  {
    sorted_window[i] = window[i];
  }
  qsort(sorted_window, window_length, sizeof(px_my_time_type), cmp_to_count_median);
  double median_value = sorted_window[window_length/2];

  free(sorted_window);
  return median_value;
}

/// @brief Функция, вычисляющая длину вектора мощностей гармоник, получающегося при применении
/// Быстрого Преобразования Фурье (FFT) к окну с заданной длинной. 
int fft_power_spectrum_length_counter(int window_length)
{
  int fft_power_spectrum_length = window_length/2 + 1;
  return fft_power_spectrum_length; 
}

/// @brief Функция, вычисляющая вектор мощностей гармоник для спектрального разложения,
/// сделанного с помощью Быстрого Преобразования Фурье (FFT). 
double* fft_power_spectrum_counter(double* window,
                                   int window_length,
                                   int power_spectrum_length)
{
  /* FFTW3 real-to-complex transform: output length = window_length/2 + 1 */
  fftw_complex *out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * power_spectrum_length);
  double *window_power_spectrum = (double*)malloc(power_spectrum_length * sizeof(double));
  if (!out || !window_power_spectrum)
  {
    if (out) fftw_free(out);
    free(window_power_spectrum);
    return NULL;
  }

  fftw_plan p = fftw_plan_dft_r2c_1d(window_length, window, out, FFTW_ESTIMATE);
  fftw_execute(p);

  for (int k = 0; k < power_spectrum_length; ++k)
  {
    double re = out[k][0];
    double im = out[k][1];
    window_power_spectrum[k] = re*re + im*im;
  }

  fftw_destroy_plan(p);
  fftw_free(out);
  return window_power_spectrum;
}

/// @brief Функция, вычисляющая евклидову норму данного вектора.
double euclidean_norm_counter(double* vector,
                              int vector_length)
{
  double euclidean_norm = 0;

  for (int i = 0; i < vector_length; i++)
  {
    euclidean_norm += vector[i]*vector[i];
  }
  euclidean_norm = sqrt(euclidean_norm);

  return euclidean_norm;
}

/// @brief Функция, вычисляющая евклидову норму вектора, полученного из данного
/// в результате исключения первого элемента вектора. 
double modified_euclidean_norm_counter(double* vector,
                                       int vector_length)
{
  double modified_euclidean_norm = 0;

  for (int i = 1; i < vector_length; i++)
  {
    modified_euclidean_norm += vector[i]*vector[i];
  }
  modified_euclidean_norm = sqrt(modified_euclidean_norm);

  return modified_euclidean_norm;
}

/// @brief Функция, вычисляющая евклидово расстояние между данными векторами. 
double euclidean_distance_counter(double* first_vector,
                                  double* second_vector,
                                  int vector_length)
{
  double euclidean_distance = 0;

  for (int i = 0; i < vector_length; i++)
  {
    euclidean_distance += (first_vector[i]-second_vector[i])*(first_vector[i]-second_vector[i]);
  }
  euclidean_distance = sqrt(euclidean_distance);
  
  return euclidean_distance;
}

/// @brief Функция, вычисляющая евклидово расстояние между векторами, полученными
/// из данных в результате исключения первых элементов данных векторов. 
double modified_euclidean_distance_counter(double* first_vector,
                                           double* second_vector,
                                           int vector_length)
{
  double modified_euclidean_distance = 0;
  
  for (int i = 1; i < vector_length; i++)
  {
    modified_euclidean_distance += (first_vector[i]-second_vector[i])*(first_vector[i]-second_vector[i]);
  }
  modified_euclidean_distance = sqrt(modified_euclidean_distance);
  
  return modified_euclidean_distance;
}

/// @brief Функция для подсчета критерия, основанного на сравнении значений скалярных величин
/// @param cur_iter Текущее количество измерений задержки (итераций).
/// @param window_selection_parameters Параметры выбора окон.
/// @param scalar_parameter_counter Указатель на функцию для подсчета скалярного параметра, характеризующего окно. 
/// @param tmp_results Вектор с текущими результатми измерения задержки (имеет длину cur_iter).
/// @return Значение критерия, основанного на сравнении значений скалярных величин.
double scalar_criterion(int cur_iter,
                        struct WindowSelectionParameters window_selection_parameters,
                        ScalarAlgorithmCounter scalar_parameter_counter,
                        px_my_time_type *tmp_results)
{
  int window_length = (int)(window_selection_parameters.ratio * (double)cur_iter);
  int max_window_begin_index = cur_iter - window_length;
  double min_max_values[2] = {-1, -1};

  for (int cur_win = 0; cur_win < window_selection_parameters.window_amount; cur_win++)
  {
    int cur_ind = rand()%max_window_begin_index;
    double cur_value_of_scalar_parameter = scalar_parameter_counter(&tmp_results[cur_ind], window_length);
    if ((min_max_values[0] == -1) || (cur_value_of_scalar_parameter < min_max_values[0]))
    {
      min_max_values[0] = cur_value_of_scalar_parameter;
    }
    if ((min_max_values[1] == -1) || (cur_value_of_scalar_parameter > min_max_values[1]))
    {
      min_max_values[1] = cur_value_of_scalar_parameter;
    }
  }
  double criterion_value = min_max_values[0]/min_max_values[1];

  return criterion_value;
}

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
                        px_my_time_type *tmp_results)
{
  int window_length = (int)(window_selection_parameters.ratio * (double)cur_iter);
  int max_window_begin_index = cur_iter - window_length;
  double sum_of_distancies = 0.0;
  double sum_of_norms = 0.0;
  double **windows_power_spectrums = (double**)malloc(window_selection_parameters.window_amount*sizeof(double*)); 
  int power_spectrum_length = counters.spectrum_length_counter(window_length);

  for (int cur_win = 0; cur_win < window_selection_parameters.window_amount; cur_win++)
  {
    int cur_ind = rand()%max_window_begin_index;
    windows_power_spectrums[cur_win] = counters.spectrum_counter(&tmp_results[cur_ind],
                                                                 window_length,
                                                                 power_spectrum_length);
    double cur_norm = counters.norm_counter(windows_power_spectrums[cur_win],
                                            power_spectrum_length);
    sum_of_norms += cur_norm;
    for (int prev_win = 0; prev_win < cur_win; prev_win++)
    {
      double cur_distance = counters.distance_counter(windows_power_spectrums[prev_win], 
                                                      windows_power_spectrums[cur_win],
                                                      power_spectrum_length);
      sum_of_distancies += cur_distance;
    }
  }
  int amount_of_distancies = window_selection_parameters.window_amount*(window_selection_parameters.window_amount-1)/2;
  double average_distance = sum_of_distancies/(double)amount_of_distancies;
  double average_norm = sum_of_norms/(double)window_selection_parameters.window_amount;
  double criterion_value = 1 - average_distance/average_norm;

  for (int i = 0; i < window_selection_parameters.window_amount; i++) {
    free(windows_power_spectrums[i]);
  }
  free(windows_power_spectrums);
  return criterion_value;
}

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
                                                  double target_criterion_value)
{
  struct AlgorithmMainInfo* algorithm_main_info = (struct AlgorithmMainInfo*)malloc(sizeof(struct AlgorithmMainInfo));
    
  if (algorithm_type == kEmptyAlgo)
  {
    algorithm_main_info->algorithm_general_type = kNoAlgo;
  }
  else {
    algorithm_main_info->frequency = frequency;
    algorithm_main_info->target_criterion_value = target_criterion_value;
    if ((ratio>0) && (ratio < 1)) {
      algorithm_main_info->window_selection_parameters.ratio = ratio;
    }
    else {
      free(algorithm_main_info);
      return NULL;
    }
    algorithm_main_info->window_selection_parameters.window_amount = window_amount;


    enum AlgorithmGeneralType cur_general_type = kScalarAlgo;
    switch (algorithm_type)
    {
    case kScalarMinAlgo:
      algorithm_main_info->scalar_algorithm_counter = min_counter;
      break;

    case kScalarAvgAlgo:
      algorithm_main_info->scalar_algorithm_counter = average_counter;
      break;
    
    case kScalarDevAlgo:
      algorithm_main_info->scalar_algorithm_counter = deviation_counter;
      break;
    
    case kScalarMedAlgo:
      algorithm_main_info->scalar_algorithm_counter = median_counter;
      break;
    
    default:
      cur_general_type = kSpectrumAlgo;
      break;
    }

    algorithm_main_info->algorithm_general_type = cur_general_type;
    if (cur_general_type == kSpectrumAlgo) {
      switch (algorithm_type)
      {
      case kSpectrumFFTAlgo:
        algorithm_main_info->spectrum_algorithm_counters.distance_counter = euclidean_distance_counter;
        algorithm_main_info->spectrum_algorithm_counters.norm_counter = euclidean_norm_counter;
        algorithm_main_info->spectrum_algorithm_counters.spectrum_counter = fft_power_spectrum_counter;
        algorithm_main_info->spectrum_algorithm_counters.spectrum_length_counter = fft_power_spectrum_length_counter;
        break;
      
      case kModifiedSpectrumFFTAlgo:
        algorithm_main_info->spectrum_algorithm_counters.distance_counter = modified_euclidean_distance_counter;
        algorithm_main_info->spectrum_algorithm_counters.norm_counter = modified_euclidean_norm_counter;
        algorithm_main_info->spectrum_algorithm_counters.spectrum_counter = fft_power_spectrum_counter;
        algorithm_main_info->spectrum_algorithm_counters.spectrum_length_counter = fft_power_spectrum_length_counter;
        break;        
      
      default:
        free(algorithm_main_info);
        algorithm_main_info = NULL;
        break;
      }
    }
  }
  return algorithm_main_info;
}