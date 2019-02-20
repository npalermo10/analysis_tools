import numpy as n
import pylab as plt
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

    def fetch_trial_data(self, *args):
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
    def __init__(self, conditions, data_dir = './'):
        
        conditions.sort(key=lambda x: x.light_num) # sort conditions based on light num
        self.conditions = conditions
        self.num_tests =  n.array([len(condition.elements) for condition in self.conditions]).prod()
        self.data_dir = data_dir
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

        self.lmr = n.zeros(n.hstack([self.d.num_trials, n.array([len(condition.elements) for condition in self.conditions]), self.trial_len]))
        self.d.set_return(start_pos=0, end_pos=self.trial_len)
        for i_coord, coord in enumerate(coords_shaped):
            slices = tuple([slice(c, c+1, None) for i_c, c in enumerate(coord)])
            try:
                self.lmr[slices] = self.d.fetch_trials_data(*mod_coords_shaped[i_coord])
            except:
                print(f"error loading lights index {mod_coords_shaped} into lmr. check that your light mods are correct and conditions are in the correct order. Is target array shape: {self.lmr.shape}?")
                
        self.lpr = n.zeros(n.hstack([self.d.num_trials, n.array([len(condition.elements) for condition in self.conditions]), self.trial_len]))
        self.d.set_return(returned_value= 'lpr', start_pos=0, end_pos=self.trial_len)
        for i_coord, coord in enumerate(coords_shaped):
            slices = tuple([slice(c, c+1, None) for i_c, c in enumerate(coord)])
            try:
                self.lpr[slices] = self.d.fetch_trials_data(*mod_coords_shaped[i_coord])
            except:
                print(f"error loading lights index {mod_coords_shaped} into lpr. check that your light mods are correct and conditions are in the correct order. Is target array shape: {self.lpr.shape}?")
