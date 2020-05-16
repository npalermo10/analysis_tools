import numpy as n
import matplotlib.pyplot as plt
import scipy.cluster.vq as clust
import os
import matplotlib.patches as mpatches
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

blue, green, yellow, orange, red, purple = [(0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37), (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]

red = [0.7686274509803922, 0.3058823529411765, 0.3215686274509804]

blue = [0.2980392156862745, 0.4470588235294118, 0.6901960784313725]

green = [0.3333333333333333, 0.6588235294117647, 0.40784313725490196]

def get_fpo_divisor(data, time_axis = -1):
    fpo_mean_expanded = n.expand_dims(data.mean(axis = time_axis), axis = time_axis).repeat(data.shape[time_axis], axis = time_axis)
    standardized_fpo_mean_expanded = fpo_mean_expanded-n.nanmin(fpo_mean_expanded) + 1
    return standardized_fpo_mean_expanded

def f_minus_i(data, i0, i1, f0,f1 = None, time_axis = None):
    if time_axis is None:
        time_axis = len(data.shape)-1
    if f1 is None:
        f1 = data.shape[time_axis]
    num_axes = len(data.shape)
    
    i_slices = [slice(None,None,None) for axis in n.arange(num_axes)]
    i_slices[time_axis] = slice(i0,i1,None)
    f_slices = [slice(None,None,None) for axis in n.arange(num_axes)]
    f_slices[time_axis] = slice(f0,f1,None)

    i_frames = data[tuple(i_slices)]
    f_frames = data[tuple(f_slices)]
    i_mean = n.expand_dims(i_frames.mean(time_axis), axis = time_axis).repeat(f_frames.shape[time_axis], axis = time_axis)

    return f_frames - i_mean

def fill_with_nans(data_list):
    num_elements = len(data_list)
    min_num_frames = min(lmr.shape[-1] for lmr in data_list)
    data_list = [lmr[...,:min_num_frames].squeeze(0) for lmr in data_list]

    list_lens = n.array([data_list[element].shape[0] for element in n.arange(num_elements)])
    amts_to_add = n.max(list_lens) - list_lens
    for i_amt_to_add, amt_to_add in enumerate(amts_to_add):
        for i in n.arange(amt_to_add):
            nan_array = n.full(n.append([1], data_list[i_amt_to_add].shape[1:]), n.nan)
            data_list[i_amt_to_add]  = n.vstack([nan_array, data_list[i_amt_to_add]]) # replace missing data
            print('added nans')
    return data_list

def fill_with_zeros(data_list):
    num_elements = len(data_list)
    min_num_frames = min(lmr.shape[-1] for lmr in data_list)
    data_list = [lmr[...,:min_num_frames].squeeze(0) for lmr in data_list]

    list_lens = n.array([data_list[element].shape[0] for element in n.arange(num_elements)])
    amts_to_add = n.max(list_lens) - list_lens
    for i_amt_to_add, amt_to_add in enumerate(amts_to_add):
        for i in n.arange(amt_to_add):
            zeros_array = n.zeros(n.append([1], data_list[i_amt_to_add].shape[1:]))
            data_list[i_amt_to_add]  = n.vstack([zeros_array, data_list[i_amt_to_add]]) # replace missing data
            print('added zeros')
    return n.array(data_list)

def reject_outliers(data, m = 3):
    medns =n.median(data, axis = 0)
    medns_exp = n.expand_dims(medns, 0).repeat(data.shape[0], axis = 0)
    d = n.abs(data - medns_exp)
    mdev = n.expand_dims(n.median(d, axis = 0), 0).repeat(data.shape[0], axis = 0)
    s = d/mdev
    data[s>m] = 0
    num_outliers = data[s>m].shape[0]
    tot_data = data.flatten().shape[0]
    print(f"{num_outliers} outliers removed out of {tot_data}")
    return data

class WBA_trial():
    '''A class to read, hold and analyze wba data from wing beat analyzers'''

    def __init__(self, data_fn, ind_chans=[2,3,4,5], nvals=2, debug=False):
        '''Read in the data'''
        self.debug = debug
        self.fn = data_fn
        self.data = n.load(data_fn)
        self.lmr = self.data[0] - self.data[1]
        self.lpr = self.data[0] + self.data[1]
        self.ind_chans = ind_chans
        self.edge_inds = {}
        self.edge_counts = {}
        for ind_chan in ind_chans:
            self.get_edge_inds(ind_chan)
            self.count_edges(ind_chan)
        bounds = self.edge_inds[ind_chans[0]][0][n.where(self.edge_inds[ind_chans[0]][1]==nvals)]
        self.starts = bounds[::2]
        self.ends = bounds[1::2]
        assert len(self.starts) == len(self.ends), "Starts amt are not equal to ends amt. Data not used."
        self.num_tests = len(self.starts)
        # self.count_flashes()
        self.set_return()

    def __repr__(self):
        return '<trial {} - {} tests>'.format(self.fn, self.num_tests)

    def set_return(self, returned_value='lmr', start_ind_chan=None, start_ind_pulse_num=0, start_pos=0, end_pos=1000):
        if returned_value == 'lmr': self.wba = self.lmr
        elif returned_value == 'lpr': self.wba = self.lpr
        if start_ind_chan == None: start_ind_chan = self.ind_chans[0] 
        self.start_ind_chan = start_ind_chan
        self.zero_inds = n.array([self.edge_inds[start_ind_chan][0][self.edge_inds[start_ind_chan][0] >= start][start_ind_pulse_num] for start in self.starts])
        self.start_pos, self.end_pos = start_pos, end_pos
        
    def __getitem__(self, *args):
        args = n.array(args).flatten()
        # for each indexing channel, where do we match the args
        locations = [self.edge_counts[self.ind_chans[i]] == args[i] for i in range(len(args))]
        # where are the matches always true
        self.locations = locations
        locations = n.where(n.product(locations, 0))
        if len(locations[0])==0: print ('no index matches')
        out = []
        for zero in self.zero_inds[locations]:
            out.append(self.wba[(zero + self.start_pos):(zero + self.end_pos)])
        return n.array(out)

    def fetch_trial_raw_data(self, channel, *args):
        args = n.array(args).flatten()
        # for each indexing channel, where do we match the args
        locations = [self.edge_counts[self.ind_chans[i]] == args[i] for i in range(len(args))]
        # where are the matches always true
        self.locations = locations
        locations = n.where(n.product(locations, 0))
        if len(locations[0])==0: print ('no index matches')
        out = []
        for zero in self.zero_inds[locations]:
            out.append(self.data[channel][(zero + self.start_pos):(zero + self.end_pos)])
        return n.array(out)

    def fetch_trial_data(self, *args):
        to_match = n.array(args).flatten()
        # for each indexing channel, where do we match the args
        locations = [self.edge_counts[self.ind_chans[i]] == to_match[i] for i in n.arange(len(to_match))]
        # where are the matches always true
        self.locations = locations
        locations = n.where(n.product(locations, 0))
        if len(locations[0])==0: print ('no index matches')
        out = []
        for zero in self.zero_inds[locations]:
            out.append(self.wba[(zero + self.start_pos):(zero + self.end_pos)])
        return n.array(out)

    def get_edge_inds(self, channel, nvals=n.array([0.5, 1.0]), duration=9, thresh=.09):
        '''Returns the indexes of rising square wave edges that last at
        least dUration samples and rise at least thresh standard
        deviations above baseline of the correlated signal.'''
        # isolate the regions above thresh stds to work with a small list of candidates
        d = self.data[channel].copy()
        d -= d.mean()
        d /= d.max()
        ptp = d.ptp()
        hi_inds = n.where(d>(d.mean() + ptp*thresh))[0]
        # find the boundaries of above thresh regions (skip more than one index)
        d_hi_inds = n.ediff1d(hi_inds, to_begin=duration, to_end=duration)
        strts = n.where(d_hi_inds>=duration)[0]
        strt_inds = n.take(hi_inds, strts[:-1])
        # one point is higher than its neighbors, return one index for each region
        peaks = n.array([d[strt_ind:strt_ind + duration].max() for strt_ind in strt_inds])
        # how best to sort into nval groups?
        centroids, vals = clust.kmeans2(peaks, nvals)
        # since the centroids are in a random order (and 1 less than I want):
        vvals = n.choose(vals, centroids.argsort() + 1)
        if not n.any(n.diff(vvals)):
            print (vvals)
            print ('xxxxx', self.fn, channel)
        # set it in the dict
        self.edge_inds[channel] = n.array([strt_inds, vvals])

    def count_edges(self, channel):
        counts = []
        was_two=False
        for val in self.edge_inds[channel][1]:
            if val == 2:
                was_two = True
            elif val == 1 and was_two:
                counts.append(1)
                was_two = False
            else:
                counts[-1] += 1
        self.edge_counts[channel] = n.array(counts)

    def count_flashes(self):
        flash_counts = []
        for channel in self.ind_chans:
            counts = []
            was_two=False
            for val in self.edge_inds[channel][1]:
                if val == 2:
                    was_two = True
                elif val == 1 and was_two:
                    counts.append(1)
                    was_two = False
                else:
                    counts[-1] += 1
            flash_counts.append(counts)
        self.flash_counts = n.array(flash_counts).T

    def response(self, test_inds, bounds=[50,100], ref_bounds=None, neg_inds=None):
        '''Return the mean response that is inside of the bounds of
        the single trace specified by test_inds. If ref_bounds are
        also specified, the mean of the trace value in them is
        subtracted out.'''
        trace = self.__getitem__(test_inds)[0]
        resp = trace[bounds[0]:bounds[1]].mean()
        ref = trace[ref_bounds[0]:ref_bounds[1]].mean() if ref_bounds else 0
        if not neg_inds:
            return resp - ref
        else:
            ntrace = self.__getitem__(neg_inds)[0]
            nresp = ntrace[bounds[0]:bounds[1]].mean()
            nref = ntrace[ref_bounds[0]:ref_bounds[1]].mean() if ref_bounds else 0
            return (resp - ref - nresp + nref)/2.
        
    def resave(self):
        os.rename(self.fn, self.fn+'x')
        n.save(self.fn, self.data)

    def fix_rep(self, chan, start, dur=80):
        first_val = self.data[chan, start]
        max_val = self.data[chan, start:start+dur].max()
        ind = start
        peak = 0
        peaked = False
        done = False
        while not peaked:
            ind += 1
            val = self.data[chan, ind]
            peak = max(val, peak)
            if val > max_val*.5 and val < peak*.9: peaked = True
        while not done:
            ind += 1
            val = self.data[chan, ind]
            if val < peak*.1 or ind-start > dur: done = True
            else: self.data[chan, ind] = first_val
        

    def plot_channel(self, chan=0, edges=True, edge_y=0):
            plt.plot(self.data[chan], 'k.-')
            if edges:
                for val in n.unique(self.edge_inds[chan][1]):
                    xs = self.edge_inds[chan][0][self.edge_inds[chan][1]==val]
                    for x in xs:
                        plt.text(x, edge_y, '{}'.format(val), va='center', ha='center')


class WBA_trials ():
    '''A class to read, hold and analyze wba data from wing beat analyzers'''

    def __init__ (self, data_dir, num_tests=8, ind_chans=[2,3,4,5], debug=False):
        '''Read in the data'''
        self.debug = debug
        fns = os.listdir(data_dir)
        self.fns = [data_dir+fn for fn in fns if fn.endswith('.npy')]
        self.fns.sort()
        self.num_trials = len(self.fns)
        self.exp_name = os.path.basename(os.path.abspath(data_dir))

        trials = []
        for fn in self.fns:
            try:
                trials.append(WBA_trial(fn, ind_chans))
                print(fn)
            except:
                print('X ' + fn + ' X')
        self.trials = [trial for trial in trials if trial.num_tests == num_tests]
        self.num_trials = len(self.trials)
        self.set_return()

    def __repr__(self):
        return '<{} - {} trials>'.format(self.exp_name, self.num_trials)

    def __len__(self):
        return len(self.trials)

    def set_return(self, returned_value='lmr', start_ind_chan=None, start_ind_pulse_num=0, start_pos=0, end_pos=1000):
        for trial in self.trials:
            if self.debug: print (trial)
            trial.set_return(returned_value, start_ind_chan, start_ind_pulse_num, start_pos, end_pos)

    def mean_sem(self, inds=[1], ninds=None, ref=None):
        out = n.array([trial.__getitem__(inds)[0] for trial in self.trials])
        if ninds:
            nout = n.array([trial.__getitem__(inds)[0] for trial in self.trials])
            # print 'dims', out.shape, nout.shape 
            out = n.vstack([out, -nout])
        if ref:
            out -= out.__getitem__(ref).mean()
        # out = n.array([trial[args] for trial in self.trials])
        return out.mean(0), out.std(0)/n.sqrt(self.num_trials)


    def __getitem__ (self, *args):
        self.a = args
        if not hasattr(args[0], '__iter__'):
            return self.trials[args[0]]
        else:
            # trials = self.trials[args[0][0]]
            trials = self.trials.__getitem__(*args[0][0:1])
            if not hasattr(trials, '__iter__'):
                trials = [trials]
            trial_args = args[0][1:]
            out = []
            for trial in trials:
                if self.debug: print ('getitem {}'.format(trial))
                out.append(trial.__getitem__(*trial_args))
            return n.array(out)

    def fetch_trials_data(self, *args):
        self.a = args
        trial = self.trials.__getitem__(self.a[0])
        trial_args = self.a[1:]
        if self.debug: print ('getitem {}'.format(trial))
        out = trial.fetch_trial_data(*trial_args)[0]
        return out

    def fetch_trials_raw_data(self, *args):
        self.a = args
        trial = self.trials.__getitem__(self.a[1])
        
        trial_args = n.hstack([self.a[0], self.a[2:]])
        if self.debug:
            print ('getitem {}'.format(trial))
            
        out = trial.fetch_trial_raw_data(*trial_args)[0]
        return out

    def response(self, test_inds=[0], bounds=[50,100],
                 ref_bounds=None, neg_test_inds=None):
        return n.array([trial.response(test_inds, bounds, ref_bounds, neg_test_inds)
                        for trial in self.trials])

    def responses(self, test_inds=[0], bounds=[50,100],
                  ref_bounds=None, neg_test_inds=None):
        # first make each element of test_inds a list, if it's not already
        t_inds = [ind if hasattr(ind, '__iter__') else [ind] for ind in test_inds]
        if neg_test_inds: #make the same list for the negative (mirrored tests)
            nt_inds = [ind if hasattr(ind, '__iter__') else [ind] for ind in neg_test_inds]
        else: #put None in each slot
            nt_inds = [[None for e in l] for l in t_inds]
        # make the output with dimension of the indexes (including size 1) and num_trials
        ind_shape = [len(ind) for ind in t_inds]     #extent of each index
        num_entries = n.product(ind_shape)           #how many total entries
        ind_inds = n.indices(ind_shape)              #indexing array into indexes
        num_dims = ind_inds.shape[0]                 #how many slots does it reference
        out = n.zeros(ind_shape + [self.num_trials]) #the output array
        # make the list to index into each entry in the output array
        # [[ii[i].flat[j] for i in range(num_dims)] for j in range(num_entries)]
        inds_list = [[ind_inds[i].flat[j] for i in range(num_dims)] for j in range(num_entries)]
        for i in range(len(inds_list)):
            inds = inds_list[i]
            resp_inds = [t_inds[d][inds[d]] for d in range(num_dims)]
            nresp_inds = [nt_inds[d][inds[d]] for d in range(num_dims)]
            out[tuple(inds)] = self.response(resp_inds, bounds, ref_bounds, nresp_inds)
        return n.squeeze(out)


class Condition():
    ''' condition used in the experiment which has a light index for each individual element'''
    def __init__(self, elements, light_num, light_mod = 0):
        self.elements = elements
        self.light_num = light_num
        self.light_mod = light_mod

class Array_builder():
    ''' builds an array given condition objects in the correct order '''
    def __init__(self, conditions, data_dir = './', raw_channels = [0,1,2,3]):
        
        conditions.sort(key=lambda x: x.light_num) # sort conditions based on light num
        self.conditions = conditions
        self.num_tests =  n.array([len(condition.elements) for condition in self.conditions]).prod()
        self.data_dir = data_dir
        self.raw_channels = raw_channels
        self.get_data()
                
    def get_data(self):
        self.d = WBA_trials(self.data_dir, self.num_tests, n.arange(len(self.conditions))+ 2)

        self.trial_len = int(n.mean([n.mean(trial.ends- trial.starts) for trial in self.d]))
        cond_el = [[i_element for i_element, element in enumerate(condition.elements)] for i_condition, condition in enumerate(self.conditions)]
        cond_el.insert(0, n.arange(self.d.num_trials))
        coords = n.stack(n.meshgrid(*cond_el), axis = len(self.conditions)+1)
        coords_shaped = coords.reshape((-1, len(self.conditions)+1))

        mod_cond_el = [[i_element + condition.light_mod for i_element, element in enumerate(condition.elements)] for i_condition, condition in enumerate(self.conditions)]
        mod_cond_el.insert(0, n.arange(self.d.num_trials))
        mod_coords = n.stack(n.meshgrid(*mod_cond_el), axis = len(self.conditions)+1)
        mod_coords_shaped = mod_coords.reshape((-1, len(self.conditions)+1))

        self.lmr = n.empty(n.hstack([self.d.num_trials, n.array([len(condition.elements) for condition in self.conditions]), self.trial_len]))
        self.lmr.fill(n.nan)
        
        self.raw_channel_data = n.array([n.zeros(n.hstack([self.d.num_trials, n.array([len(condition.elements) for condition in self.conditions]), self.trial_len])) for channel in self.raw_channels])
        self.d.set_return(start_pos=0, end_pos=self.trial_len)
        for i_coord, coord in enumerate(coords_shaped):
            slices = tuple([slice(c, c+1, None) for i_c, c in enumerate(coord)])
            try:
                self.lmr[slices] = self.d.fetch_trials_data(*mod_coords_shaped[i_coord])
                
            except:
                print(f"error loading coordinate: {coord} into lights index {mod_coords_shaped[i_coord]} into lmr. check that your light mods are correct and conditions are in the correct order. Is target array shape: {self.lmr.shape}? If not there might be a problem with file: {self.d[coord[0]]}")
            try:
                for i_channel, channel in enumerate(self.raw_channels):
                    self.raw_channel_data[i_channel][slices] = self.d.fetch_trials_raw_data(channel, *mod_coords_shaped[i_coord])

            except:
                print(f'error importing raw data channel')
        self.lpr = n.empty(n.hstack([self.d.num_trials, n.array([len(condition.elements) for condition in self.conditions]), self.trial_len]))
        self.lpr.fill(n.nan)
                
        self.d.set_return(returned_value= 'lpr', start_pos=0, end_pos=self.trial_len)
        for i_coord, coord in enumerate(coords_shaped):
            slices = tuple([slice(c, c+1, None) for i_c, c in enumerate(coord)])
            try:
                self.lpr[slices] = self.d.fetch_trials_data(*mod_coords_shaped[i_coord])
            except:
                print(f"error loading coordinate: {coord} into lights index {mod_coords_shaped[i_coord]} into lpr. check that your light mods are correct and conditions are in the correct order. Is target array shape: {self.lpr.shape}? If not there might be a problem with file: {self.d[coord[0]]}")
            

class Data_handler():
    ''' this class gets the mean and std err for data. Takes either data array or list. Also performs hasty stats.'''
    def __init__(self, data, trial_axis = 0, time_axis = -1):
        self.data = data
        self.trial_axis = trial_axis
        self.time_axis = time_axis
        self.is_list = False
        if isinstance(self.data, list):
            self.is_list = True ##handling of lists is different than
        self.flat_mean = [] # mean along time axis
        self.flat_se = [] # se along time axis
        self.mean = [] # mean of entire time series
        self.se = [] # se of entire time searies

    def calc_means_se(self):
        if self.is_list:
            flat_means = []
            flat_ses = []
            means = []
            ses = []
            for d in self.data:
                means.append(d.mean(axis = self.trial_axis))
                se.append(d.std(axis = self.trial_axis))
        else:
            self.mean = self.data.mean(axis = trial_axis)
            self.se = self.data.std(axis = trial_axis)/sqrt(shape(self.data)[trial_axis])
            self.flat_mean = self.data.mean(axis = time_axis)
            
class Hasty_plotter():
    ''' this class should speed up common tasks such as displaying every plot or means of all the plots. It is not intended to be for final production analyzing. If you want to put in data that is already time averaged, then just make sure you expand_dims on it so that it has a time axis which is size 1. '''
    def __init__(self, data, trial_axis = 0,  time_axis = None, plot_title= None, starting_fig_num = 0, color_axis = None, color_labels = None, color_list = None, subplot_axis = None, subplot_labels = None,  x_axis = None, figure_axis = None,  legend_title = None, start_t = 0, end_t = 1, x_vals = None, x_ticks = None, x_label = None, y_axis = None, y_label = None,  y_vals = None, y_ticks = None, rm_outliers = False):
        assert len(data.shape) >= 3, 'Data must be at least 3 dimensions to plot.'
        self.time_axis = time_axis
        self.data = data
        
        self.trial_axis = trial_axis
        self.color_axis = color_axis
        self.color_labels = color_labels
        self.color_list = color_list
        self.x_axis = x_axis
        self.x_vals = x_vals
        self.x_ticks = x_ticks
        self.x_label = x_label
        self.y_axis = y_axis
        self.y_label = y_label
        self.y_vals = y_vals
        self.y_ticks = y_ticks
        self.subplot_axis = subplot_axis
        self.subplot_labels = subplot_labels
        self.figure_axis = figure_axis
        self.legend_title = legend_title
        self.start_t = start_t
        self.end_t = end_t
        if time_axis == None:
            self.time_axis = len(data.shape) -1
        self.plot_title = plot_title
        self.rm_outliers = rm_outliers
        self.num_trials = self.data.shape[trial_axis]
        self.frames = self.data.shape[self.time_axis]
        self.starting_fig_num = starting_fig_num # so you can make a new hasty plotter object that won't override figs from another
        self.figs = []

    def update_axes_info(self, **kwargs):
        orig_axes_info = {'subplot_axis' : self.subplot_axis,
                          'subplot_labels' : self.subplot_labels,
                          'color_axis' : self.color_axis,
                          'color_labels' : self.color_labels,
                          'x_axis' : self.x_axis,
                          'x_label': self.x_label,
        }
        for kwarg in kwargs:
            if kwarg == 'subplot_axis':
                if 'subplot_labels' not in kwargs:
                    if kwargs['subplot_axis'] == orig_axes_info['color_axis']:
                        self.subplot_labels = orig_axes_info['color_labels']
                    if kwargs['subplot_axis'] == orig_axes_info['x_axis']:
                        self.subplot_labels = orig_axes_info['x_label']
                self.subplot_axis = kwargs['subplot_axis']
            if kwarg == 'color_axis':
                if 'color_labels' not in kwargs:
                    if kwargs['color_axis'] == orig_axes_info['subplot_axis']:
                        self.color_labels = orig_axes_info['subplot_labels']
                    if kwargs['color_axis'] == orig_axes_info['x_axis']:
                        self.color_labels = orig_axes_info['x_label']
                self.color_axis = kwargs['color_axis']
            if kwarg == 'x_axis':
                if 'x_label' not in kwargs:
                    if kwargs['x_axis'] == orig_axes_info['subplot_axis']:
                        self.x_labels = orig_axes_info['subplot_labels']
                    if kwargs['x_axis'] == orig_axes_info['color_axis']:
                        self.x_labels = orig_axes_info['color_labels']
                self.x_axis = kwargs['x_axis']
                    
    
    def eq_ylims(self, fig_ind = 0, y_max= None, y_min = None):
        if y_max is None:
            y_max = max(ax.get_ylim()[1] for ax in self.figs[fig_ind].axes)
        if y_min is None:
            y_min = min(ax.get_ylim()[0] for ax in self.figs[fig_ind].axes)
    
        for ax in self.figs[fig_ind].axes:
            ax.set_ylim([y_min, y_max])
            
    def plot_time_series(self, **kwargs):
        self.update_axes_info(**kwargs)
        subplot_axis = self.subplot_axis
        subplot_labels = self.subplot_labels
        color_axis = self.color_axis
        color_labels = self.color_labels
        trial_axis = self.trial_axis
        time_axis = self.time_axis
        x_axis = self.x_axis
        
        assert self.frames > 1, f"You can't plot time series data if you have no time series. You only have {self.frames} frame of data."
        if x_axis is not None and subplot_axis is None:
            subplot_axis = x_axis
        num_axes_in_args = (color_axis is not None) +  (subplot_axis is not None)
        assert len(self.data.shape) == 2 + num_axes_in_args, f'Incorrect number of axes arguments. There should only be {len(self.data.shape)} and you have {num_axes_in_args}'
        fig = plt.figure(len(self.figs) + self.starting_fig_num)
        self.figs.append(fig)
        data = self.data
        if not subplot_axis:
            subplot_axis = len(data.shape)
            data = n.expand_dims(data, axis = -1)
            
        if not color_axis:
            color_axis = len(data.shape)
            data = n.expand_dims(data, axis = -1)            

        num_subplots = data.shape[subplot_axis]
        num_colors = data.shape[color_axis]
        
        data = data.transpose(trial_axis, subplot_axis, color_axis, time_axis)
        
        mean = n.nanmean(data, axis = 0)
        sd_err = n.nanstd(data, axis = 0)/n.sqrt(data.shape[0])
        
        fig.suptitle(f'{self.plot_title} - {self.data.shape[trial_axis]} flies')    
        for plot_num in n.arange(num_subplots):
            
            ax = fig.add_subplot(num_subplots, 1, plot_num + 1)
            plt.axhline(0, color = 'k', linestyle = '--')
            plt.axvline(0, color = 'k', linestyle = '--')
            if subplot_labels is not None:
                ax.set_title(subplot_labels[plot_num])
            for color in n.arange(num_colors):
                mean2plot = n.squeeze(mean[plot_num, color])
                std_err2plot = n.squeeze(sd_err[plot_num, color])
                plt.plot(mean2plot)
                plt.fill_between(n.arange(int(self.frames*self.end_t) - int(self.frames*self.start_t)), mean2plot + std_err2plot,  mean2plot- std_err2plot, alpha = 0.3)
                if color_labels is not None:
                    patches =[mpatches.Patch(color = "C" + str(color), label = str(color_labels[color])) for color in n.arange(num_colors)]
                    plt.legend(title = self.legend_title, handles=patches)
        
    def plot_mean_resp(self, save_fig= False, save_name="plot", **kwargs):
        self.update_axes_info(**kwargs)
        subplot_axis = self.subplot_axis
        color_axis = self.color_axis
        trial_axis = self.trial_axis
        x_axis = self.x_axis
        time_axis = self.time_axis

        if subplot_axis is None and self.subplot_axis is not None:
            subplot_axis = self.subplot_axis

        if color_axis is None and self.color_axis is not None:
            color_axis = self.color_axis

        if x_axis is None and self.x_axis is not None:
            x_axis = self.x_axis
            
        num_axes_in_args = (color_axis is not None or self.color_axis is not None) + (x_axis is not None or self.x_axis is not None) + (subplot_axis is not None or self.subplot_axis is not None)
        assert len(self.data.shape) == num_axes_in_args + 2, f'Incorrect number of axes arguments. There should only be {len(self.data.shape)} and you have {num_axes_in_args}'
        fig = plt.figure(len(self.figs) + self.starting_fig_num)
        self.figs.append(fig)
        data = self.data
        
        if subplot_axis is None:
            subplot_axis = len(data.shape) 
            data = n.expand_dims(data, axis = -1)
            
        if color_axis is None:
            color_axis = len(data.shape)
            data = n.expand_dims(data, axis = -1)            

        if  x_axis is None:
            x_axis = len(data.shape)
            data = n.expand_dims(data, axis = -1)            

        num_subplots = data.shape[subplot_axis]
        num_colors = data.shape[color_axis]
        num_xs = data.shape[x_axis]
        if self.x_vals is not None and self.x_ticks is None:
            self.x_ticks = self.x_vals
            
        if self.x_vals is None:
            self.x_vals = n.arange(num_xs)
        
        data = data.transpose(self.trial_axis, subplot_axis, color_axis, x_axis, self.time_axis)
        data_means_over_t = n.nanmean(data[...,int(self.start_t*self.frames): int(self.end_t*self.frames)], axis = 4)
        if self.rm_outliers:
            data_means_over_t = reject_outliers(data_means_over_t)
            
        mean = n.nanmean(data_means_over_t, axis = 0)
        sd_err = n.nanstd(data_means_over_t, axis= 0)/n.sqrt(data_means_over_t.shape[trial_axis])
         
        plt.suptitle(f'{self.plot_title} - {self.data.shape[self.trial_axis]} flies')    
        for plot_num in n.arange(num_subplots):
            ax = plt.subplot(num_subplots, 1, plot_num + 1)
            if self.subplot_labels:
                ax.title.set_text(self.subplot_labels[plot_num])
            plt.axhline(0, color = 'k', linestyle = '--')
            plt.axvline(0, color = 'k', linestyle = '--')
            offset = 0.00
            if self.x_ticks is not None:
                plt.xticks(self.x_vals, self.x_ticks)
            plt.xlabel(self.x_label)
            plt.ylabel(self.y_label)
            colors = ["C" + str(color) for color in n.arange(num_colors)]
            if self.color_list is not None:
                colors = self.color_list
            if self.color_labels is not None:
                patches =[mpatches.Patch(color = colors[color], label = str(self.color_labels[color])) for color in n.arange(num_colors)]
                plt.legend(title = self.legend_title, handles=patches)
            for color in n.arange(num_colors):
                plt.errorbar(self.x_vals + offset, mean[plot_num, color], yerr = sd_err[plot_num, color], marker = 'o', ms = 9.0, color = colors[color])
                offset += (self.x_vals[1:] - self.x_vals[:-1]).mean()/num_xs*0.2
        
        if save_fig:
            save_name = save_name + ".svg"
            plt.tight_layout()
            plt.savefig(save_name, format = "svg")
        plt.show(block=False)
                
    def plot_mean_resp_heatmap(self,  center_zero = False, cmap = 'viridis', **kwargs):
        self.update_axes_info(**kwargs)
        subplot_axis = self.subplot_axis
        color_axis = self.color_axis
        trial_axis = self.trial_axis
        x_axis = self.x_axis
        time_axis = self.time_axis
        y_axis = self.y_axis
        y_ticks = self.y_ticks
        y_labels = self.y_label
        
        assert x_axis, "can't plot heatmap without x axis"
        if y_axis is None and color_axis is not None:
            y_axis = color_axis
            if y_ticks is None:
                y_ticks = self.color_labels
        assert y_axis, "can't plot heatmap without y axis"
        fig = plt.figure(len(self.figs) + self.starting_fig_num)
        self.figs.append(fig)
        data = []
                    
        if subplot_axis is not None:
            data = self.data.transpose(trial_axis, subplot_axis, y_axis, x_axis, time_axis)
        else:
            subplot_axis = len(self.data.shape)
            data = n.expand_dims(self.data, axis = -1)
            data = data.transpose(trial_axis, subplot_axis, y_axis, x_axis, time_axis)
        mean = data.mean(axis = 0)[..., self.frames*self.start_t:self.frames*self.end_t].mean(axis = -1)    
        num_subplots = mean.shape[0]    
        
        plot_max = mean.max()
        plot_min = mean.min()
        if center_zero:
            plot_max = n.abs(mean).max()
            plot_min = -plot_max

        plt.suptitle(f'{self.plot_title} - {self.data.shape[self.trial_axis]} flies')
        for plot_num in n.arange(num_subplots):
            ax = plt.subplot(num_subplots, 1, plot_num + 1)
            if self.subplot_labels is not None:
                ax.set_title(str(self.subplot_labels[plot_num-1]))
            img = plt.imshow(mean[plot_num-1], cmap = cmap, vmin = plot_min, vmax = plot_max)
            plt.colorbar(img, cmap = cmap)
            if self.x_ticks is None:
                pass
            else:
                ticks = n.arange(self.data.shape[x_axis])
                plt.xticks(ticks, self.x_ticks)
            if y_ticks is None:
                pass
            else:
                ticks = n.arange(self.data.shape[y_axis])
                plt.yticks(ticks, y_ticks[::-1])


            if self.x_label:
               plt.xlabel(self.x_label) 
            if y_label:
               plt.ylabel(y_label)
        
    def plot_indv_mean_resps(self, regression = False, **kwargs):
        self.update_axes_info(**kwargs)
        subplot_axis = self.subplot_axis
        color_axis = self.color_axis
        trial_axis = self.trial_axis
        x_axis = self.x_axis
        time_axis = self.time_axis
        color_labels = self.color_labels
        
        num_axes_in_args = (color_axis is not None) + (x_axis is not None) + (subplot_axis is not None)
        assert len(self.data.shape) == num_axes_in_args + 2, f'Incorrect number of axes arguments. There should only be {len(self.data.shape)} and you have {num_axes_in_args}'
        fig = plt.figure(len(self.figs) + self.starting_fig_num)
        self.figs.append(fig)
        data = self.data
        if subplot_axis is None and self.subplot_axis is None:
            subplot_axis = len(data.shape)
            data = n.expand_dims(data, axis = -1)
            
        if color_axis is None and self.color_axis is None:
            color_axis = len(data.shape)
            data = n.expand_dims(data, axis = -1)            

        if x_axis is None and self.x_axis is None:
            color_axis = len(data.shape)
            data = n.expand_dims(data, axis = -1)            

        num_subplots = data.shape[subplot_axis]
        num_colors = data.shape[color_axis]
        len_x_axis = data.shape[x_axis]

        if self.x_vals is not None and self.x_ticks is None:
            self.x_ticks = self.x_vals
            
        if self.x_vals is None:
            self.x_vals = n.arange(num_xs)

        data = data.transpose(self.trial_axis, subplot_axis, color_axis, x_axis, self.time_axis)
        mean = data[..., self.frames*self.start_t:self.frames*self.end_t].mean(axis = 4)
        if self.rm_outliers:
            mean = reject_outliers(mean)
        plt.suptitle(f'{self.plot_title} - {self.data.shape[self.trial_axis]} flies')    
        for plot_num in n.arange(num_subplots):
            ax = plt.subplot(num_subplots, 1, plot_num + 1)
            if self.subplot_labels:
                ax.title.set_text(self.subplot_labels[plot_num])
            plt.axhline(0, color = 'k', linestyle = '--')
            plt.axvline(0, color = 'k', linestyle = '--')
            offset = 0.02
            plt.xticks(n.arange(len_x_axis), self.x_ticks)
            plt.xlabel(self.x_label)
            if color_labels is not None:
                patches =[mpatches.Patch(color = "C" + str(color), label = str(color_labels[color])) for color in n.arange(num_colors)]
                plt.legend(title = self.legend_title, handles=patches)
            offset = 0    
            for color in n.arange(num_colors):
                
                if regression:
                    xs = n.hstack(n.array([self.x_vals]*mean.shape[0]))
                    ys = n.hstack(mean[:, plot_num, color])
                    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
                    plt.plot(xs, intercept + slope*xs, 'r', label='fitted line', color = "C" + str(color), linewidth =2.0)
                   
                for trial in mean:
                    rgb_colors = n.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
                    rgb_color = n.hstack([rgb_colors[color], [0.65]])
                    plt.scatter(self.x_vals + offset , trial[plot_num, color], color = rgb_color, s = 8)
                offset += 0.1    
        
