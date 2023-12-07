def meas_discharge_time1 (time, data, start_val, stop_val):
	t_start = meas_when (time, data, start_val)
	t_stop = meas_when (time, data, stop_val)
	t_diff = t_stop - t_start
	return t_diff

def meas_discharge_time2 (time, data, start_val, t_stop):
	t_start = meas_when (time, data, start_val)
	t_diff = t_stop - t_start
	return t_diff

def meas_when (time, value, point):
	value_ = [ abs( item - point ) for item in value ]
	minpos = value_.index(min(value_))
#	print (minpos)
	return time[minpos]

def plotting3d (x, y, z):
	ax = plt.figure().add_subplot(projection='3d')
	ax.scatter(x, y, z, linewidth=0.2, antialiased=True)

