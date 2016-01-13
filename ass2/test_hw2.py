import timeit
import hw2

# timing
time_coinresult = timeit.timeit(stmt='hw2.coin_result(10)',setup='import hw2', number=1000)

print(time_coinresult)

time_trial = timeit.timeit(stmt='hw2.trial(1000,10)',setup='import hw2', number=1000)

print(time_trial)
