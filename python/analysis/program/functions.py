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

def exp2model(e):
    # C[i] * X^n * Y^m
    return ' + '.join([
        f'C[{i}]' +
        ('*' if x>0 or y>0 else '') +
        (f'X^{x}' if x>1 else 'X' if x==1 else '') +
        ('*' if x>0 and y>0 else '') +
        (f'Y^{y}' if y>1 else 'Y' if y==1 else '')
        for i,(x,y) in enumerate(e)
    ])

def plotting2d (x, y):
    fig, ax = plt.subplots()
    plot_y = ax.plot(x, y, 'o')
    plt.grid (visible=True, which='major', axis='both')
#
