#include "Statistics.h"

double dev_const(int n) {
	return static_cast<double>(n) - 1 / static_cast<double> (n * n);
}

double mean(vector <double> const &arr) {
	double result = 0.0;
	for (int i = 0; i < arr.size(); i++) {
		result += arr[i];
	}
	result /= static_cast<double>(arr.size());
	return result;
}

double std_dev(vector <double> const &arr) {
	double result = 0.0;
	double mn = mean(arr);
	for (int i = 0; i < arr.size(); i++) {
		result += (arr[i] - mn) * (arr[i] - mn);
	}
	result /= static_cast<double>(arr.size());
	result = sqrt(result);
	return result;
}

double median(vector <double> a) {
	for (int i = 0; i < a.size(); i++) {
		for (int j = i; j < a.size(); j++) {
			if (a[i] > a[j]) {
				double sw = a[i];
				a[i] = a[j];
				a[j] = sw;
			}
		}
	}
	return a[a.size() / 2];
}

void median_filter(vector <double> &arr) {
	vector <double> result(arr.size());
	for (int i = 0; i < arr.size(); i++) {
		vector <double> window(3);
		window[1] = arr[i];
		if (i == 0) {
			window[0] = 0;
			window[2] = arr[i + 1];
		} 
		else if (i == arr.size() - 1) {
			window[0] = arr[i - 1];
			window[2] = 0;
		}
		else {
			window[0] = arr[i - 1];
			window[2] = arr[i + 1];
		}
		result[i] = median(window);
	}
	arr = result;
}

void trend_estimination(vector <double> const &arr, double &slope, double &intercept) {
	double mean_x = 0.0, mean_y = 0.0;
	for (int i = 0; i < arr.size(); i++) {
		mean_x += static_cast<double> (i);
	}
	mean_x /= arr.size();
	mean_y = mean(arr);
	double sq1 = 0.0, sq2 = 0.0;
	for (int i = 0; i < arr.size(); i++) {
		sq1 += (static_cast<double>(i) - mean_x) * (arr[i] - mean_y);
		sq2 += (static_cast<double>(i) - mean_x) * (static_cast<double>(i) - mean_x);
	}
	slope = sq1 / sq2;
	intercept = mean_y - slope * mean_x;
}

vector <double> detrend(vector <double> const &arr) {
	double m, b;
	trend_estimination(arr, m, b);
	vector <double> result(arr.size());
	for (int i = 0; i < arr.size(); i++) {
		result[i] = arr[i] - (m * static_cast<double>(i) + b);
	}
	return result;

}

vector <vector <vector <double>>> corr_peaks(ClustData &clData) {
	vector <vector <vector <double>>> corr_result;
	int num_proc = clData.getInfo().proc_num;
	int mes_total = (clData.getInfo().end_mes_len - clData.getInfo().begin_mes_len) / clData.getInfo().step_len;
	vector <vector <ClustData::elem>> cl_data = clData.getClData();
	corr_result.resize(num_proc);
	for (int i = 0; i < num_proc; i++) {
		corr_result[i].resize(num_proc);
		for (int j = 0; j < num_proc; j++) {
			corr_result[i][j].resize(mes_total);
		}
	}
	for (int i = 0; i < num_proc; i++)
		for (int j = 0; j < num_proc; j++) {
			corr_result[i][j] = detrend(cl_data[i][j].med);
		}
	for (int i = 0; i < num_proc; i++)
		for (int j = 0; j < num_proc; j++) {
			double sdev = std_dev(corr_result[i][j]);
			double mn = mean(corr_result[i][j]);
			if (sdev <= 0)
				sdev = 1;
			for (int k = 0; k < mes_total; k++) {
				corr_result[i][j][k] = (corr_result[i][j][k] - mn) / (sdev * sqrt(dev_const(mes_total - 1)));
			}
		}
	return corr_result;
}

vector <vector <vector <double>>> corr_jumps(ClustData &clData) {
	vector <vector <vector <double>>> corr_result;
	int num_proc = clData.getInfo().proc_num;
	int mes_total = (clData.getInfo().end_mes_len - clData.getInfo().begin_mes_len) / clData.getInfo().step_len;
	vector <vector <ClustData::elem>> cl_data = clData.getClData();
	corr_result.resize(num_proc);
	for (int i = 0; i < num_proc; i++) {
		corr_result[i].resize(num_proc);
		for (int j = 0; j < num_proc; j++) {
			corr_result[i][j].resize(mes_total);
		}
	}
	for (int i = 0; i < num_proc; i++)
		for (int j = 0; j < num_proc; j++) {
			median_filter(cl_data[i][j].med);
			for (int k = 0; k < mes_total - 1; k++) {
				corr_result[i][j][k] = cl_data[i][j].med[k + 1] - cl_data[i][j].med[k];
			}
			corr_result[i][j][mes_total - 1] = corr_result[i][j][mes_total - 2];
			corr_result[i][j][0] = corr_result[i][j][1];
		}

	for (int i = 0; i < num_proc; i++)
		for (int j = 0; j < num_proc; j++) {
			double sdev = std_dev(corr_result[i][j]);
			double mn = mean(corr_result[i][j]);
			if (sdev <= 0)
				sdev = 1;
			for (int k = 0; k < mes_total; k++) {
				corr_result[i][j][k] = (corr_result[i][j][k] - mn) / (sdev * sqrt(dev_const(mes_total - 1)));
			}
		}
	return corr_result;
}