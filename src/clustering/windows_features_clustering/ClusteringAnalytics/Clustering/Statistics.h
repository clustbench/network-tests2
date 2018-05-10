#pragma once
#include <vector>
#include <cmath>
#include "ClusterData.h"

using namespace std;

double dev_const(int n);
double mean(vector <double> const &arr);
double std_dev(vector <double> const &arr);
void trend_estimination(vector <double> const &arr, double &slope, double &intercept);
vector <double> detrend(vector <double> const &arr);
void median_filter(vector <double> &arr);
vector <vector <vector <double>>> corr_peaks(ClustData &clData);
vector <vector <vector <double>>> corr_jumps(ClustData &clData);