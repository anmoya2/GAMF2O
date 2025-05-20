import os
import numpy as np
import random
import math
import time

from sklearn.linear_model import LinearRegression




class Genetic_HB:
    def __init__(self, type_attributes, ranges, n_at, model_wrapper, train_X, train_Y_one_hot, val_X, val_Y_one_hot, test_X, test_Y_one_hot, out_path,
        train_config = None, include_acc = False, generations=12, n_ind = 4, n_cut=4, delta=3.0, epsilon=0.0005,
        seed = 1234, cut_el = 2.0, adaptive_w=True, per_to_pend = 0.5,
        cut_point_options = None, type_m=1, type_c=1, cuts=1, mut_percent = 0.05,
         hof_size = 2, w_init = 0.4, f_enf = None, options_mut = None):

        """
        init function of our genetic idea.

        :param type attributes: type of attributes for creating gens of this type.
        :param ranges: ranges in which the parameters are included.
        :param n_at: total length of the individual.
        :param model_wrapper: Wrapper of a DL model
        :param train_X: Dataset for training (data)
        :param train_Y_one_hot: Dataset for training (labels in one hot encoding)
        :param val_X: Dataset for validation (data)
        :param val_Y_one_hot: Dataset for validation (labels in one hot encoding)
        :param test_X: Dataset for testing (data)
        :param test_Y_one_hot: Dataset for testing (labels in one hot encoding)
        :param out_path: Output path
        :param train_config configuration for training
        :param include_acc: Should the individual include the accuracy value?(Classification)
        :param generations: Maximum generations of the algorithm.
        :param n_ind: number of individuals of the population.
        :param n_cut: number of points of time in which the GA has to decide the individuals to continue training.
        :param delta: delta that controls the number of resources (epochs) that are assigned at each step.
        :param epsilon: Threshold for convergence
        :param seed: random seed
        :param cut_el: value that controls the number of individuals that continue training at each step.
        :param adaptive_w: indicates if the value for loss and slope are fixed or adaptive.
        :param per_to_pend: amount of epochs to have into account for calculating the slope
        :param cut_point_options: points of the individual that can be used for cutting.
        :param type_m: type of mutation. Option 1 indicates mutate by gen, option 2, by blocks.
        :param type_c: type of cross. Only option 1 is programmed (by block), but it is open for the future.
        :param cuts: Number of cuts for creating children.
        :param mut_percent: percentage of mutation.
        :param hof_size: size of the elite group.
        :param w_init: Between loss and the slope, this indicates the weight for the loss.
        :param f_enf: cooling function
        :param options_mut: we offer different options of mutation (3) that allow more or less exploration.
        """ 

        self._n_ep_total = 0

        self._n_cut = n_cut
        self._delta = delta
        if n_ind % 2 != 0:
            raise Exception("Population size has to be even")
        self._n_ind = n_ind
        self._best = None
        self._c_pop = []
        self.out_path = out_path
        self._n_of_pairs = int(self._n_ind / 4)
        self._c_child_pop = []
        self._seed = seed
        self._n_generations = generations
        self._n_at = n_at
        self._type_attributes  = type_attributes
        self._ranges = ranges
        self._rng = random.Random(self._seed)
        self._cut_el = cut_el
        self._w_init = w_init
        self._adaptive_w = adaptive_w
        self._per_to_pend = per_to_pend
        #self._types_of_gen = types_of_gen
        self._cut_point_options = cut_point_options
        self._type_m = type_m
        self._type_c = type_c
        self._cuts = cuts
        self._mut_percent = mut_percent
        self._hof_size = hof_size
        self._hof = None
        self.INF = 100000
        self._f_enf = f_enf
        w_loss_without_norm = [self._f_enf(x, self._w_init) for x in range(self._n_cut)]
        min_el = np.amin(np.asarray(w_loss_without_norm))
        max_el = np.amax(np.asarray(w_loss_without_norm))
        self._w_loss = [((x - min_el)/(max_el - min_el))*(1 - self._w_init) + self._w_init for x in w_loss_without_norm]
        self._w_slope = [1-x for x in self._w_loss]
        self._options_mut = options_mut
        self._old_best_loss = None
        self._num_gen_without_imp = 0

        self._model_wrapper = model_wrapper
        self.train_X, self.train_Y_one_hot = train_X, train_Y_one_hot
        self.val_X, self.val_Y_one_hot = val_X, val_Y_one_hot
        self.test_X, self.test_Y_one_hot = test_X, test_Y_one_hot
        self._include_acc=include_acc
        self._epsilon = epsilon
        self.train_config = train_config




    def _generate_pop(self):
        """
        function for generating the population.
        """ 
        for _ in range(self._n_ind):
            if self._include_acc:
                self._c_pop.append({"ind": self._generate_random_ind(), "val_loss": None, 
                "slope": None, "model": None, "weights": None, "F": None, "test_loss": None, "test_acc":None})
            else:
                self._c_pop.append({"ind": self._generate_random_ind(), "val_loss": None, 
                "slope": None, "model": None, "weights": None, "F": None, "test_loss": None})
    def _generate_random_ind(self):
        """
        function for generating an individual.
        :return an individual.
        """

        ind = []
        for t, ranges in zip(self._type_attributes, self._ranges):
            if t == 'int':
                ind.append(self._generate_int(ranges))
            elif t == 'float':
                ind.append(self._generate_float(ranges))
            else:
                ind.append(self._generate_cat(ranges))
        return np.asarray(ind)

    
    def _generate_int(self, ranges):
        """
        Function for generating an integer in a range.
        :param ranges: ranges for generating the number.
        :return an integer
        """
        min_v, max_v = ranges
        return self._rng.randint(min_v, max_v)
    def _generate_float(self, ranges):
        """
        Function for generating a float in a range.
        :param ranges: ranges for generating the number.
        :return a float
        """
        min_v, max_v = ranges
        return self._rng.uniform(min_v, max_v)


    def _generate_cat(self, list_op):
        """
        Function for randomly choosing an element of a list.
        :param list_op: list of options.
        :return an option of the list.
        """
        return self._rng.choice(list_op)


    def _mutate(self, indiv, type_m, index):
        """
        Function for the mutation.

        :param indiv: the individual to mute.
        :param type_m: Type of mutation to follow. Option 1 change just a gen and option 2 all gens of a block.
        :param index: index (of gen or block) to change.
        :return the individual mutated.
        """
        

        if type_m == 1:
            t = self._type_attributes[index]
            if t == "int" or t == "float":
                min_v, max_v = self._ranges[index]

                self._change_ind_t_1(indiv, t, index, min_v, max_v)
            else:
                self._change_ind_t_1(indiv, t, index, options = self._ranges[index],)

        elif type_m == 2:
            self._change_ind_t_2(indiv, index)
        else:
            raise Exception("Type of mutation is not valid")


        return indiv

    def _change_ind_t_1(self,el, t, index, min_v= None, max_v = None, options = None):
        """
        function for change a index following type 1.

        :param el indicates the individual.
        :param index indicates the index to change.
        :param min_v, max_v, options indicate the ranges/list of options for the new value.
        """
        if t == 'int':
            #different element. Lazy selection
            value = self._rng.randint(min_v, max_v)
            while(value == el[index]):
                value = self._rng.randint(min_v, max_v)
            el[index] = value

        elif t == 'float':
            #different element. Lazy selection
            value = self._rng.uniform(min_v, max_v)
            while(value == el[index]):
                value = self._rng.uniform(min_v, max_v)
            el[index] = value
        else:
            val_options = np.asarray([opt for opt in options if opt != el[index]])
            el[index] = self._rng.choice(options)




    def _change_ind_t_2(self, el, index):
        """
        function for change a index following type 2.

        :param el indicates the individual.
        :param index indicates the index to change.
        """
        
        indexes_to_change = []

        if index == 0:
            indexes_to_change = np.arange(self._cut_point_options[0])
            
        
        #last block
        elif index == len(self._cut_point_options):
            indexes_to_change = np.arange(self._cut_point_options[-1], self._n_at)

        elif index>0 and index < len(self._cut_point_options):
            indexes_to_change = np.arange(self._cut_point_options[index-1], self._cut_point_options[index])



        else:
            raise Exception("Index indicated is not related to a valid block")


        for ind_to_c in indexes_to_change:
            self._mutate(el, 1, ind_to_c)


    def _select(self, k=0.3):
        """
        function for selecting possible parents (based on tournament).

        :param k proportion of population to take for the parents.
        :return the selected pairs of parents to cross and mutate.
        """
        t_size = round(k*self._n_ind)
        t_size = t_size if t_size>=2 else 2
        n_of_pairs = self._n_of_pairs 
        selected_pairs = []
        for _ in range(n_of_pairs):
            parent1, parent2 = self._select_pair(t_size)
            #new: have to check if new pair is inside

            while ((parent1, parent2) in selected_pairs or (parent2, parent1) in selected_pairs):
                parent1, parent2 = self._select_pair(t_size)

            selected_pairs.append((parent1, parent2))
        return selected_pairs

    def _select_pair(self, t_size):
        """
        function for selecting a pair of parents from a tournament of size t_size.

        :param t_size size of the tournament.
        :return pair of parents.
        """

        #not let equal parents
        indexes = [i for i in range(self._n_ind)]
        selected_indexes = self._rng.sample(indexes, t_size)
        '''
        first_p = np.amin(np.asarray(selected_indexes))
        indexes_2 = [i for i in range(self._n_ind) if i!= first_p]

        selected_indexes_2 = self._rng.sample(indexes_2, t_size)
        two_p = np.amin(np.asarray(selected_indexes_2))
        '''
        first_p, two_p = np.sort(np.asarray(selected_indexes))[0:2]
        return first_p, two_p




       

    def _validate(self, mut):
        """
        Function for checking (or fixing) a new individual
        :param mut is the individual.

        return the new individual fixed.
        """
        val_mut = []
        for el, t, ranges in zip(mut, self._type_attributes, self._ranges):
            if t == "int" or t == "float":
                min_r, max_r = ranges
                if el < min_r:
                    val_mut.append(min_r)
                elif el > max_r:
                    val_mut.append(max_r)
                else:
                    val_mut.append(el)
            else:
                val_mut.append(el)
        return np.asarray(val_mut)

    def _swap(self, arr1, arr2 ,i,j):
        """
        Function to swap two arrays

        :param arr1, arr2 are the individuals
        :param i,j start and end for swapping
        """
        aux = (arr1[i:j]).copy()
        
        arr1[i:j] = arr2[i:j].copy()
        
        arr2[i:j] = aux

    def _cross(self, el, el2):
        """
        function for crossing two individuals

        :param el, el2 are the individuals to cross.
        :return two children obtained from the cross.
        """
        indiv1 = el["ind"]
        indiv2 = el2["ind"]

        if self._cuts > len(self._cut_point_options):
            raise Exception("Sorry, cuts should be equal/less than len of ranges gen")
        child_1 = np.copy(indiv1)
        child_2 = np.copy(indiv2)
        


        #type 1: superficial change --- > Main idea. To add more
        if self._type_c == 1:
            cut_points = self._rng.sample(self._cut_point_options, self._cuts)
            change_el = True
            for i in range(self._cuts+1):

                #change alternatively
                if change_el == True:
                    if i == 0:
                        min_inx = 0
                        max_inx = cut_points[i]
                        
                    elif i == self._cuts:
                        min_inx = cut_points[i-1]
                        max_inx = self._n_at

                    else:
                        min_inx = cut_points[i-1]
                        max_inx = cut_points[i]


                    self._swap(child_1, child_2, min_inx, max_inx)
                    

                    change_el = False
                else:
                    change_el = True

        if self._include_acc:
            c_1 = {"ind": child_1, "val_loss": None, "slope": None, 
                    "model": None, "weights": None, "F": None, "test_loss": None, "test_acc": None}
            c_2 = {"ind": child_2, "val_loss": None, "slope": None, 
                    "model": None, "weights": None, "F": None, "test_loss": None, "test_acc":None}
        else:
            c_1 = {"ind": child_1, "val_loss": None, "slope": None, 
                    "model": None, "weights": None, "F": None, "test_loss": None}
            c_2 = {"ind": child_2, "val_loss": None, "slope": None, 
                    "model": None, "weights": None, "F": None, "test_loss": None}
        return c_1, c_2

        

    def _evaluate_ind(self, generation, cut, i, indiv, batch_size, epochs, type_pop = "pop"):
        """
        function for evaluating and individual until a fidelity (number of epochs).

        :param geneartion is used for indicating the generation and properly save the individual.
        :param cut is used for indicating in which step of the iteration (cut point) we are currently.
        :param i number of the individual in the population (used for creating the name)
        :param indiv is the individual
        :param batch_size Batch size
        :param epochs is the number of epochs to run the DL model.
        :param type_pop indicates the type of pop we are evaluating (used for the name)
        :return the individual evaluated
        """
        

        ind = indiv["ind"]
        model_load = indiv["model"]

        
        self._model_wrapper.build_model(ind)

        if model_load != None:
            self._model_wrapper.load(model_load)

    

        
        n_ep = int(epochs)

        if n_ep < 2:
            n_ep = 2
       
        
        m_name = self.out_path + "/genetic-" + type_pop + '-model_'+str(generation)+'_'+str(i)+'_'+str(cut)+'.h5'

        w_name = self.out_path + "/genetic-weights-" + type_pop + '-model_'+str(generation)+'_'+str(i)+'_'+str(cut)+'.h5'
       
        #train_config = {"epochs": n_ep, "batch_size":batch_size, "callbacks": callbacks,
        #"checkpoint_path": m_name, "verbose": 0}

        self.train_config["epochs"]=n_ep
        self.train_config["checkpoint_path"] = m_name
        

        historytemp = self._model_wrapper.train(self.train_X, self.train_Y_one_hot, self.val_X, self.val_Y_one_hot ,self.train_config)
        


        history_loss = np.asarray(historytemp.history['val_loss'])
        
        self._n_ep_total = self._n_ep_total + n_ep
        val_coef = 0
        tam_history_loss = len(history_loss)
        if tam_history_loss >= 2:
        
            to_calculate_pend = int(self._per_to_pend * tam_history_loss) if int(self._per_to_pend * tam_history_loss)>=2 else 2
            arange = np.arange(to_calculate_pend).reshape((-1,1))
            model_reg = LinearRegression()
            model_reg.fit(arange, history_loss[-1*to_calculate_pend:])
            slope = model_reg.coef_[0]
            if slope > 0:
                slope = 0
            val_coef = slope 


        val_loss = np.amin(history_loss)

        model_to_ind = None
        w_to_ind = None

        #Case no improvement in new evaluations
        if indiv["val_loss"]!=None and val_loss >= indiv["val_loss"]:
            val_loss = indiv["val_loss"]
            val_coef = 0
            model_to_ind = indiv["model"]
            w_to_ind = indiv["weights"]
        else:
            model_to_ind = m_name
            w_to_ind = w_name
            self._model_wrapper.save(model_to_ind)

        
        
        if self._include_acc:
            test_loss, test_acc = self._model_wrapper.evaluate(self.test_X, self.test_Y_one_hot)
        
            ind_ev = {'ind': ind, 'val_loss': val_loss, 'slope': val_coef, "model": model_to_ind,
            "weights": w_to_ind, "F": None, "test loss": test_loss, "test_acc": test_acc}
        else:
            test_loss = self._model_wrapper.evaluate(self.test_X, self.test_Y_one_hot)
        
            ind_ev = {'ind': ind, 'val_loss': val_loss, 'slope': val_coef, "model": model_to_ind,
            "weights": w_to_ind, "F": None, "test loss": test_loss}


        return ind_ev

    
    def _calculate_epochs(self, max_epochs, i):
        """
        function for calculate the epochs to evaluate in step i.
        :param max_epochs indicates the maximum number of epochs in total.
        :param i indicates in which step we are.
        :return the number of epochs to evaluate in step i
        """

        return max_epochs/(self._delta**i)

    def _calculate_epochs_array(self, max_epochs):
        """
        function for indicating the epochs in each step.
        :param max_epochs indicates the maximum number of epochs in total.
        :return an array with the number of epochs (resources) in each step.
        """
        n_epochs = np.asarray([self._calculate_epochs(max_epochs, i) 
            for i in range(self._n_cut-1, -1, -1)])

        for i in range(len(n_epochs)):
            n_epochs[i] = int(n_epochs[i])
            
                
            if i == 1:
                
                n_epochs[i] = int(n_epochs[i] - n_epochs[i-1])

            elif i>1:
                n_epochs[i] = int(n_epochs[i] - np.sum(n_epochs[:i]))
            #at least 2 epochs
            if n_epochs[i] <2:
                n_epochs[i] = 2
        return n_epochs

    def _calculate_n_el(self, max_el, i):
        """
        function for calculate the individual to continue training in step i.
        :param max_el indicates the maximum number of individuals in total.
        :param i indicates in which step we are.
        :return the number of individuals to evaluate in step i
        """

        n = int(max_el/(self._cut_el**i))
        if n < 1:
            n = 1
        return n

    def _eval_process(self, generation, batch_size, n_epochs, n_el_to_ev, pop, type_pop = "pop"):
        """
        function for evaluating all the population until the end.

        :param geneartion is used for indicating the generation and properly save the individual.
        :param batch_size indicates the batch_size for evaluating
        :param n_epochs indicates the number of epochs
        :param n_el_to_ev indicates the number of individuals to ev at each step
        :param pop is the current pop
        :param type_pop indicates the type of pop we are evaluating (used for the name)
        :return the population evaluated
        """
        for i in range(len(n_el_to_ev)):
            #not last cut
            if i != len(n_el_to_ev) - 1:
                #regarding name of models, only in first pass create a new name
                #Lazy nomenclature
                min_loss = self.INF
                max_loss = -1* self.INF

                min_slope = self.INF
                max_slope = -1 * self.INF
                for j in range(n_el_to_ev[i]):
                    pop[j] = self._evaluate_ind(generation, i, j, pop[j], batch_size, n_epochs[i],type_pop)
                    if pop[j]["val_loss"] < min_loss:
                        min_loss = pop[j]["val_loss"]
                    if pop[j]["val_loss"] > max_loss:
                        max_loss = pop[j]["val_loss"]
                    if pop[j]["slope"] < min_slope:
                        min_slope = pop[j]["slope"]
                    if pop[j]["slope"] > max_slope:
                        max_slope = pop[j]["slope"]

                #Not evaluated, poor ind
                for el in pop[n_el_to_ev[i]:]:
                    el["F"] = self.INF
                



                #new ev
                for el in pop[:n_el_to_ev[i]]:
                    #If not last cut
                    if max_loss - min_loss == 0:
                        loss_norm = 0
                    else:

                        loss_norm = (el["val_loss"] - min_loss)/(max_loss - min_loss)
                    if max_slope - min_slope == 0:
                        slope_norm = 0
                    else:
                        slope_norm = (el["slope"] - min_slope)/(max_slope - min_slope)
                    
                    el["F"] = self._w_loss[i] * loss_norm + self._w_slope[i] * slope_norm
                

            else:
                #Last evaluation
                for j in range(n_el_to_ev[i]):
                    pop[j] = self._evaluate_ind(generation, i, j, pop[j], batch_size, n_epochs[i], type_pop)
                for el in pop[n_el_to_ev[i]:]:
                    el["F"] = self.INF

                for el in pop[:n_el_to_ev[i]]:
                    el["F"] = el["val_loss"]


            pop.sort(key=lambda a: (a["F"], a["val_loss"]))


        return pop

    def _mutate_by_index(self, index_of_ind, index_changed, type_m = 1):
        """
        function for changing an index that it is not in the list of previously changed.

        :param index_of_ind index of the individual to access to its list of indexed previously mutated.
        :param index_changed list of indexes changed in all the individuals
        :param type_m type of mutation
        :return index to change.
        """
        if type_m == 1:
            index_to_c = self._rng.choice([k for k in range(self._n_at) if k not 
                            in index_changed[index_of_ind]])
        elif type_m == 2:
            index_to_c = self._rng.choice([k for k in range(len(self._cut_point_options)+1) if k not 
                            in index_changed[index_of_ind]])
        
        mut = self._mutate(self._c_child_pop[index_of_ind]["ind"], type_m, index = index_to_c)
        
        self._c_child_pop[index_of_ind]["ind"] = mut
        return index_to_c

    def _is_better_than(self, ind1, ind2):
        """
        function that controls if ind2 is better than ind1 or not.
        """
        is_better = False
        if (ind1["F"] < ind2["F"]) or (ind1["F"] == ind2["F"] and ind1["val_loss"]<ind2["val_loss"]):
            is_better = True
        return is_better

    def _ind_in_current_pop(self, ind):
        """
        function that controls if the individual ind is or not in the current population.
        """
        ind_inside = False
        for el in self._c_pop:

            if (el["ind"] == ind["ind"]).all():
                ind_inside = True
                break

        return ind_inside

    def _actualize_c_pop(self):
        """
        Function for actualizing the current pop with the new children pop (elitist)
        """
        self._c_pop.extend(self._c_child_pop[0:self._hof_size])

        self._c_pop.sort(key=lambda a: (a["F"], a["val_loss"]))
        self._c_pop = self._c_pop[0:self._n_ind]



    def _first_run(self, batch_size, n_epochs, n_el_to_ev, n_el_to_ev_child):
        """
        function for running the first generation.
        :param batch_size indicates the batch size for evaluating the model
        :param n_epochs indicates the number of epochs
        :param n_el_to_ev indicates the number of elements to evaluate from parent and offspring populations.
           """
        self._c_pop = self._eval_process(0, batch_size, n_epochs, n_el_to_ev,
         self._c_pop)
        self._hof = self._c_pop[0:self._hof_size]
        self._best = self._hof[0]

        print("self best: ", self._best)

        


        selected_t = self._select()
        for tup in selected_t:
            index1, index2 = tup
            el = self._c_pop[index1]
            el2 = self._c_pop[index2]
            crossed1, crossed2 = self._cross(el, el2)
            self._c_child_pop.append(crossed1)
            self._c_child_pop.append(crossed2)

        n_to_mut = None
        max_gen = None


        if self._options_mut != None:
            self._type_m = self._options_mut[0]["type_m"]
            self._mut_percent = self._options_mut[0]["mut_percent"]

        tam_c_pop = len(self._c_pop)

        if self._type_m == 1:
            n_to_mut = round(tam_c_pop*self._n_at*self._mut_percent)
            max_gen = self._n_at

        elif self._type_m == 2:
            n_to_mut = round(tam_c_pop*(len(self._cut_point_options)+1)*self._mut_percent)
            max_gen = len(self._cut_point_options)+1


        len_child = len(self._c_child_pop)
        n_to_mut = n_to_mut if n_to_mut>=1 else 1
        gen_by_ind = int(n_to_mut/len_child)
        gen_extra = n_to_mut % len_child
        #Change gen_by_ind per individual
        index_changed = [[] for _ in range(len_child)]
        for _ in range(gen_by_ind):
            
            for j in range(len_child):
                #check if all have been changed
                if len(index_changed[j]) == max_gen:
                    break
                else:
                    index_to_c = self._mutate_by_index(j, index_changed, self._type_m)
                    index_changed[j].append(index_to_c)
                
        #change 1 gen of some individuals
        extra_to_c = self._rng.sample([i for i in range(len_child)], gen_extra)
        for iid in extra_to_c:
            index_to_c = self._mutate_by_index(iid, index_changed, self._type_m)
            index_changed[iid].append(index_to_c)






        self._c_child_pop = self._eval_process(0, batch_size, n_epochs, n_el_to_ev_child, 
            self._c_child_pop, type_pop = "child_pop")

                
        #next generation: pop + hof

        
        self._actualize_c_pop()


        
        mypath = self.out_path
        allfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        self._hof = self._c_pop[0:self._hof_size]
        self._best = self._hof[0]

        best_model_to = self._best["model"]
        best_w_to = self._best["weights"]
        #Remove unnecesary
        for f in allfiles:
            complete_name = self.out_path+"/"+f
            if best_model_to != complete_name and best_w_to != complete_name:
                if os.path.exists(complete_name):
                    os.remove(complete_name)
       
        self._c_child_pop = []

    def run_evol(self, batch_size, max_epochs):
        """
        function for running all the algorithm.
        :param batch_size indicates the batch size.
        :param max_epochs indicates the maximum of epochs for training.
           """
        t1 = time.time()
        #First generation
        self._generate_pop()

        
        n_epochs = self._calculate_epochs_array(max_epochs)
        n_el_to_ev = np.asarray([self._calculate_n_el(self._n_ind, i) for i
            in range(self._n_cut) ])

        n_el_to_ev_child = np.asarray([self._calculate_n_el(int(self._n_ind/2), i) for i
            in range(self._n_cut) ])
        #epochs at each step. To complet n epochs with m yet trained, we need n-m

        #GEN 0

        self._first_run(batch_size, n_epochs, n_el_to_ev, n_el_to_ev_child)

        option_mut_to_choice = 0
        old_best_losses = [h["val_loss"] for h in self._hof]
        old_best_loss_avg = sum(old_best_losses)/self._hof_size


        generet = 1
        end = False
        while end==False:
            selected_t = self._select()
            for tup in selected_t:
                index1, index2 = tup
                el = self._c_pop[index1]
                el2 = self._c_pop[index2]
                crossed1, crossed2 = self._cross(el, el2)
                self._c_child_pop.append(crossed1)
                self._c_child_pop.append(crossed2)
        
            n_to_mut = None
            max_gen = None

            if self._options_mut != None:
                self._type_m = self._options_mut[option_mut_to_choice]["type_m"]
                self._mut_percent = self._options_mut[option_mut_to_choice]["mut_percent"]
                    

            


            if self._type_m == 1:
                n_to_mut = round(self._n_ind*self._n_at*self._mut_percent)
                max_gen = self._n_at
            #Changing n_to_mut
            elif self._type_m == 2:
                n_to_mut = round(self._n_ind*(len(self._cut_point_options)+1)*self._mut_percent)
                max_gen = len(self._cut_point_options)+1



            len_child = len(self._c_child_pop)
            n_to_mut = n_to_mut if n_to_mut>=1 else 1
            gen_by_ind = int(n_to_mut/len_child)
            gen_extra = n_to_mut % len_child
            #Change gen_by_ind per individual
            index_changed = [[] for _ in range(len_child)]
            for _ in range(gen_by_ind):
                
                for j in range(len_child):
                    #check if all have been changed
                    if len(index_changed[j]) == max_gen:
                        break
                    else:
                        index_to_c = self._mutate_by_index(j, index_changed, self._type_m)
                        index_changed[j].append(index_to_c)
                    
            #change 1 gen of some individuals
            extra_to_c = self._rng.sample([i for i in range(len_child)], gen_extra)
            for iid in extra_to_c:
                index_to_c = self._mutate_by_index(iid, index_changed, self._type_m)
                index_changed[iid].append(index_to_c)



            self._c_child_pop = self._eval_process(generet, batch_size, n_epochs,
             n_el_to_ev_child, self._c_child_pop, type_pop = "child")

            
            #next generation: pop + hof

            
            
            self._actualize_c_pop()
            
            self._hof = self._c_pop[0:self._hof_size]


            self._best = self._hof[0]
            self._c_child_pop = []

            mypath = self.out_path
            allfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
            best_model_to = self._best["model"]
            best_w_to = self._best["weights"]
            for f in allfiles:
                complete_name = self.out_path+"/"+f
                if best_model_to != complete_name and best_w_to != complete_name:
                    if os.path.exists(complete_name):
                        os.remove(complete_name)

            new_best_losses = [h["val_loss"] for h in self._hof] 
            new_best_loss_avg = sum(new_best_losses)/self._hof_size
            epsilon = self._epsilon
            
            if new_best_loss_avg >= (old_best_loss_avg - epsilon):
                self._num_gen_without_imp = self._num_gen_without_imp + 1
            else:
                old_best_loss_avg = new_best_loss_avg
                self._num_gen_without_imp = 0
            #3 generation without improvement
            if self._num_gen_without_imp < 4:
                option_mut_to_choice = 0
            #4 or 5 or 6 generation without improvement
            elif self._num_gen_without_imp >=4 and self._num_gen_without_imp < 7:
                option_mut_to_choice = 1
            #7 or 8 or 9 generation without improvement
            elif self._num_gen_without_imp >= 7 and self._num_gen_without_imp < 10:
                option_mut_to_choice = 2
            else:
                end = True
                print("generation: ", generet, " best: ", self._hof)
                print("Number of epochs: ", self._n_ep_total)


                break

            
            print("generation: ", generet, " best: ", self._hof)
            print("Number of epochs: ", self._n_ep_total)

            generet = generet + 1
        t2 = time.time()
        print("time: ", t2-t1)
            



    