# -*- coding: utf-8 -*-

import numpy
from matplotlib import pyplot
from scipy.optimize import curve_fit 

#Asymmetry function
def A0(t, B, beta,Tau):
    """Used for modelling aasymmetry ratio between left and right muon detection channels.
    Returns an exponenially decaying oscillating function over some time array t,
    for given inputs for B, beta, Tau"""
    gamma = 851.616 #10E6 factor removed because x-axis of plot is scaled down by 10E6
    return ((-1.0/3.0)*((numpy.sin(gamma*B*t - beta)-numpy.sin(gamma*B*t + beta))/(2.0*beta)))*numpy.exp(-t/Tau)

#Quadratic for 
def Quadratic(E, a, b, c):
    """Used for modelling magnetic field dependence on implantation energy.
    Returns a quadratic function over a set of energy values E."""
    return a*E*E + b*E + c 

def SortData(filename):
    """Processes raw data; sorts each implantation energy into left and right channels.
    Returns two arrays of data:
    Channel1Data contains left detector times for 5,10,15,20,25KeV energies (eg.5keV times are in Channel1Data[0]).
    Channel2Data contains right detector times."""
    
    #Check correct file type has been input
    assert (filename[-4:-1] != ".dat"),"Input file must be .dat type"
    
    f = open(filename, 'r')
    i=1
    # counting how many lines until header break "&END" is reached.
    # Needed for "skip header" option in genfromtxt
    for line in f:
        i=i+1
        if "&END" in line:
            break
    
    data = numpy.genfromtxt(filename, delimiter = '\t', skip_header=i);
    #Check correct data file is being used
    assert (len(data[0]) == 10),"Not enough implantation energies in data. Five required (5,10,15,20,25keV)."

    Channel1Data = [];
    Channel2Data = [];
    i = 0
    j = 0
    #Loop over each data column to create two arrays, Ch1 and Ch2, of decay times for each energy
    while i <9:
        Channel1Data.append(data[data[:,i+1]==1,i]);
        Channel2Data.append(data[data[:,i+1]==2,i]);
        j+=1
        i+=2
    
    
    return [Channel1Data,Channel2Data]

def Histograms(Ch1, Ch2):
    """Visualises Ch1 and Ch2 data as histograms with 500 bins each.
    Arguments are single arrays."""
    #Channel 1 histogram
    pyplot.hist(Ch1, bins=500, color='b')
    pyplot.title('py15csa - Histogram - 10keV Left Channel Decays')
    pyplot.xlabel('Decay Time ' '('r'$\mu$''s)')
    pyplot.ylabel('Number of Decays')
    pyplot.show()
    #Channel 2 histogram
    pyplot.hist(Ch2, bins=500, color='b')
    pyplot.title('py15csa - Histogram - 10keV Right Channel Decays')
    pyplot.xlabel('Decay Time ' '('r'$\mu$''s)')
    pyplot.ylabel('Number of Decays')
    pyplot.show()

    
def Asymmetry(Ch1,Ch2):
    """Takes input data for a given energy, combines values to find raw Asymmetry data
    and analyses asymmetry to give estimates for magnetic field (B) and beta values
    which can be used in a curve_fit function.
    Returns:
    B_guess, beta_guess, A0data (asymmetry ratio), A0error"""
    Bguess = []
    betaguess = []
    A0_error = []
    A0_Data = []
    for i in range(len(Ch1)):
        
        #Turning Ch1 and Ch2 data into bins of equal size
        timesCh1, bins1 = numpy.histogram(Ch1[i],bins=400)
        timesCh2, bins2 = numpy.histogram(Ch2[i],bins=400)
        
        #Converting the int values in the time arrays to float to allow for division in next step
        timesCh1  = timesCh1.astype(numpy.float) 
        timesCh2  = timesCh2.astype(numpy.float)
        
        #Combining Ch1 and Ch2 data to find asymmetry
        A0Data = (timesCh1-timesCh2)/(timesCh1+timesCh2)
        
        #Errors in Ch1, Ch2 and A0 data
        Ch1error = numpy.sqrt(timesCh1)
        Ch2error = numpy.sqrt(timesCh2)
        A0error = 2.0*(numpy.sqrt((1.0/(numpy.power(timesCh1 + timesCh2,4.0)))*(numpy.power(timesCh1,2.0)*numpy.power(Ch1error,2.0)+numpy.power(timesCh2,2.0)*numpy.power(Ch2error,2.0))))
        
        #Guess B value using Fourier transform to find frequency
        
        fourier = numpy.fft.fft(A0Data)
        fourier = numpy.abs(fourier)
        n = fourier.size
        freq = numpy.fft.fftfreq(n)
        
        #array slot containing max frequency
        maxx = numpy.argmax(fourier)
        freq_guess = numpy.abs(freq[maxx])
        #convert to numerical value based on the 400 bins used
        B_guess = (freq_guess/400.0)*100.0
        
        #Find period (not in seconds - measured in 'number of array values per oscillation')
        period = int(1.0/freq_guess)
        half_period = period/2
        
        #Guess beta value based on raw Asymmetry data
        maxx = numpy.argmax(A0Data[0:half_period])
        #Take numerical value of largest data point in first half-period and convert to beta-value      
        beta_guess = numpy.abs(2.0*A0Data[maxx])
        
        Bguess.append(B_guess)
        betaguess.append(beta_guess)
        A0_error.append(A0error)
        A0_Data.append(A0Data)
        
    return [Bguess, betaguess, A0_error, A0_Data]
    
def GetFitData(B_guess, beta_guess, A0error, A0Data):  
    """Takes raw Asymmetry data and guesses for B, beta and Tau, and puts into 
    curve_fit to give final outputs for what B, beta and Tau should be in Asymmetry function.
    Returns:
    B, beta, Tau, B_error, beta_error, Tau_error."""
    #x-axis points for raw data in curve_fit, in microseconds
    B = []
    beta = []
    Tau = []
    B_error = []
    beta_error = []
    Tau_error = []
    for i in range(len(B_guess)):
        A0times = numpy.linspace(0,10,400)         
        #raw data curve_fit and errors in B and beta
        popt,pcov = curve_fit(A0, A0times, A0Data[i],p0=(B_guess[i],beta_guess[i],4.0), sigma=A0error[i], absolute_sigma=True)
        errors = numpy.sqrt(numpy.diag(pcov))
        
        #If Tau value is too high it means the data is probably undamped 
        assert(numpy.abs(popt[2])<30.0),"Damping value appears vastly incorrect. This could mean the input data is undamped. Check input data has damping turned on."
        
        #return B, beta, Tau, B_error, beta_error, Tau_error for ProcessData results
        B.append(popt[0])
        beta.append(popt[1])
        Tau.append(popt[2])
        B_error.append(errors[0])
        beta_error.append(errors[1])
        Tau_error.append(errors[2])
        
    return [B, beta, Tau, B_error, beta_error, Tau_error]
    
    
def PlotA0Data(B, B_error, beta, beta_error, Tau, Tau_error, A0error, A0Data):
    """Plots fitted Asymmetry curve against raw Asymmetry data points
    Outpues plot = "10keV Asymmetry Data"."""
    #new x-axis points for plotting curve_fit 
    new_time = numpy.linspace(0,10,1000)
    
    #x-axis for plotting raw A0 data (400 bins)
    A0times = numpy.linspace(0,10,400)
    
    #new A0 points using B and beta values from GetFitData
    new_A0=A0(new_time,B, beta, Tau)
    
    fig = pyplot.figure()
    axes = fig.add_subplot(111)
    axes.set_title('py15csa - 10keV Asymmetry Data')
    axes.set_xlabel('Time ' '('r'$\mu$''s)')
    axes.set_ylabel('Asymmetry')
    axes.errorbar(A0times,A0Data,yerr=A0error, fmt="bo", label='Raw data')
    axes.plot(new_time, new_A0,"g-", label = "Fitted plot")
    axes.legend(loc=1)
    #y-axis defined by max and min values of A0Data being plotted
    axes.axis([0,10,-A0Data[0]-0.5*A0Data[0],A0Data[0]+0.5*A0Data[0]])
    axes.text(1,-A0Data[0]-0.2*A0Data[0],   r'$B = %.3f \pm %.3f mT, $'
                                            r'$\beta = %.2f \pm %.2f rad, $'
                                            r'$\tau_{damp} = %.1f \pm %.1f \mu s$'
                                            %(B*1000,B_error*1000,beta,beta_error,Tau,Tau_error))
    pyplot.show()

def Field_B_Variation(B, B_error):
    """Takes array of B values and their errors and inputs them into curve_fit
    for Quadratic function.
    Finds values a, b, c to fit to Quadratic, B = aE^2+bE+c .
    Uses values a, b, c to plot fitted curve against raw data points for B.
    Outputs plot - "Magnetic Field Variation with Implantation Energy".
    Returns a, b, c"""
    #curve_fit x-axis for energies from 5 to 25KeV, for plotting raw data
    energies = numpy.linspace(5,25,5)
    #curve_fit for finding a, b, c in Quadratic plot
    popt,pcov = curve_fit(Quadratic,energies,B,sigma = B_error)
    errors = numpy.sqrt(numpy.diag(pcov))
    
    #Assign a, b, c and their errors
    a = popt[0]
    b = popt[1]
    c = popt[2]
    a_error = errors[0]
    b_error = errors[1]
    c_error = errors[2]
    
    #new x-axis for fitted data
    new_energies = numpy.linspace(5,25,100)
    
    #Use a, b, c values from curve fit to creating fitted plot
    fitted_curve = Quadratic(new_energies,a,b,c)
    
    fig = pyplot.figure()
    axes = fig.add_subplot(111)
    axes.set_title('py15csa - Magnetic Field Variation\n with Implantation Energy')
    axes.set_xlabel('Implantation energy 'r'$E$'' (KeV)')
    axes.set_ylabel('Magnetic field ' r'$B=\mu_0 H$'' (T)')
    axes.errorbar(energies, B, yerr=B_error, fmt="bo", label = "Measured magnetic field")
    axes.plot(new_energies,fitted_curve,"g-", label = "Fitted plot")
    axes.legend(loc=4)
    #y-axis defined by limits of data being plotted 
    axes.axis([5,25,B[0],B[4]+B[4]/10.0])
    axes.text(15,B[1]-B[1]/10.0, r'$B= aE^{2}+bE+c$'
                      '\n'r'$a= %.2f \pm %.2f \mu TkeV^{-2}$'
                        '\n'r'$b= %.2f \pm %.2f \mu TkeV^{-1}$'
                        '\n'r'$c= %.2f \pm %.2f \mu T$'
                            #*1000000 for better presentation - avoids 0.000001 sorts of values
                        %(a*1000000,a_error*1000000,b*1000000,b_error*1000000,c*1000000,c_error*1000000))
    pyplot.show()

    
    #return a, b, c and their errors
    return [a, b, c, a_error, b_error, c_error]


def ProcessData(filename):
    """Takes a given file of muon detection data for processing and analysis.
    Input file should be .dat and contain data sets for 5 implantation energies, 5,10,15,20,25keV,
    and be tab-delineated.
    Outputs:
    A histogram of raw count data for left detection channel for 10keV implantation energy.
    A histogram of raw count data for right detection channel for 10keV implantation energy.
    Plot of raw asymmetry data with fitted curve for 10keV implantation energy.
    Plot of Magnetic field data point calculations with fitted curve
    to show how field changes with implantation energy @ 10keV.
    
    Returns:
    Dictionary of results: B, B_error, beta, beta_error, Tau, Tau_error,
    Energy Coeffecients (a, b, c), Coeffienct errors(a_error,b_error,c_error)."""
    
    print 'Got file...: "%s"\n' %(filename)
    print "Processing raw data...\n"    
    #Sort input file into times for Ch1 and Ch2 data
    Ch1Data,Ch2Data = SortData(filename)
    
    print "Making histograms of Ch1 and Ch2 data @ 10keV...\n"
    #Make histograms for Ch1 and Ch2 data for 10KeV energy
    Histograms(Ch1Data[1],Ch2Data[1])
    
    print "Estimating values for B, beta, and Tau...\n"
    #Create asymmetry data and find adequate guesses and error values to put into curve_fit function
    B_guess, beta_guess, A0error, A0Data = Asymmetry(Ch1Data,Ch2Data)

    #Put guesses into curve_fit to bring out optimal values for B, beta, Tau, B_error, beta_error and Tau_error
    B_, beta_, Tau_, B_error_, beta_error_, Tau_error_ = GetFitData(B_guess, beta_guess, A0error, A0Data)
      
    print "Plotting Asymmetry data @ 10keV...\n"
    #Plot curve_fit data and raw data with errors for 10KeV energy
    PlotA0Data(B_[1], B_error_[1], beta_[1], beta_error_[1], Tau_[1], Tau_error_[1], A0error[1], A0Data[1])
    
    print "Plotting magnetic field variation with implantation energy...\n"

    #Curvit fit and plots for B variation with energy
    a, b, c, a_error, b_error, c_error = Field_B_Variation(B_, B_error_)
    
    #Need to scale Tau values down (they were scaled up for ease of fitting A0 plot to time axis)
    Tau10k = Tau_[1]*1E-06
    Tau_error10k= Tau_error_[1]*1E-06
    
    print "Final results...\n"
    
    results={"10keV_B":B_[1], #Magnetic field for 10keV data (T)
             "10keV_B_error":B_error_[1], # Error in the magnetic field (T)
             "beta": beta_[1], #Detector angle in radians
             "beta_error": beta_error_[1], #Uncertainity in detector angle (rad)
             "10keV_tau_damp": Tau10k, #Damping time for 10keV (s)
             "10keV_tau_damp_error": Tau_error10k, #Tau error (s)
             "B(Energy)_coeffs":(a,b,c), #tuple of a,b,c for quadratic,linear and constant terms
                                                  #for fitting B dependence on energy
                                                  #(T/keV^2,T/keV,T)
             "B(Energy)_coeffs_errors":(a_error,b_error,c_error), # Errors in above in same order.
             }
    return results

if __name__=="__main__":
    
    filename="assessment_data_damping_multiple_py15csa.dat"
    test_results=ProcessData(filename)
    print test_results