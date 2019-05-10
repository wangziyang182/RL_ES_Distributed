import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ars_mc_data =np.array(pd.read_csv('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/ARS_MountainCarContinuous-v0_perturbations_64_state_renormalize_True_Simga_0.3_best_64_test_0.csv')['avg_fit'])
vanilla_mc_data = np.array(pd.read_csv('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/Vanilla_MountainCarContinuous-v0_perturbations_64_state_renormalize_False_Simga_0.3_best_32_test_0data.csv')['avg_fit'])
cdpp_mc_data = np.array(pd.read_csv('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/CDPP_MountainCarContinuous-v0_perturbations_64_state_renormalize_True_Sigma_0.3_best_64_test_0data.csv')['avg_fit'])[:-1]
rank_mc_data = np.array(pd.read_csv('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/Rank_MountainCarContinuous-v0_perturbations_64_state_renormalize_True_Simga_0.3_best_64_test_0data.csv',header=2)['avg_fit'].dropna())
cma_mc_data = np.array(pd.read_excel('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/CMA_ContinuousMountainCar_sigma_1_num_perturbations_128.xlsx')).flatten()


print(ars_mc_data.shape)
print(vanilla_mc_data.shape)
print(cdpp_mc_data.shape)
print(rank_mc_data.shape)
print(cma_mc_data.shape)

x = np.arange(200)

plt.figure()
plt.title('ES Methods on MountainCarContinuous_v1 Environment')
plt.plot(x,ars_mc_data[:-1],label='ARS')
plt.plot(x,vanilla_mc_data,label='Vanilla ES')
plt.plot(x,cdpp_mc_data,label='ES with CDPP')
plt.plot(x,rank_mc_data,label='ES with Rank')
plt.plot(x,cma_mc_data,label='CMA-ES')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('reward')
plt.show()

plt.figure()
plt.title('ES Methods on MountainCarContinuous_v1 Environment')
plt.plot(x[25:],ars_mc_data[25:-1],label='ARS')
plt.plot(x[25:],vanilla_mc_data[25:],label='Vanilla ES')
plt.plot(x[25:],cdpp_mc_data[25:],label='ES with CDPP')
plt.plot(x[25:],rank_mc_data[25:],label='ES with Rank')
plt.plot(x[25:],cma_mc_data[25:],label='CMA-ES')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('reward')
plt.show()



ars_swimmer_data =np.array(pd.read_excel('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/ARS_Swimmer-v2_Rank_perturbations_64_state_renormalize_True_Simga_0.1_best_64_test_0data.xlsx',header=1)['avg_fit'])
vanilla_swimmer_data = np.array(pd.read_csv('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/Vanilla_Swimmer-v2_perturbations_64_state_renormalize_False_Simga_0.3_best_32_test_3data.csv')['avg_fit'])
cdpp_swimmer_data = np.array(pd.read_csv('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/CDPP_Swimmer-v2_perturbations_64_state_renormalize_False_Sigma_0.3_best_64_test_2data.csv')['avg_fit'])
rank_swimmer_data = np.array(pd.read_csv('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/Rank_Swimmer-v2_perturbations_64_state_renormalize_False_Simga_0.3_best_32_test_0data.csv')['avg_fit'].dropna())
cma_swimmer_data = np.array(pd.read_csv('/Users/joshrutta/Desktop/Spring 2019/Big_Data_and_Machine_Learning/plotting_data/CMA_Swimmer-v2_perturbations_64_state_renormalize_False_Sigma_1_best_64_test_1data.csv')['avg_fit'].dropna()).flatten()

# vanilla_swimmer_data = np.append(vanilla_swimmer_data,np.zeros(301))
# cdpp_swimmer_data = np.append(vanilla_swimmer_data,np.zeros(301))
# rank_swimmer_data = np.append(vanilla_swimmer_data,np.zeros(301))
# cma_swimmer_data = np.append(vanilla_swimmer_data,np.zeros(301))
# print()

# print(ars_swimmer_data.shape)
# print(vanilla_swimmer_data[:200].shape)
# print(cdpp_swimmer_data[:200].shape)
# print(rank_swimmer_data[:200].shape)
# print(cma_swimmer_data[:200].shape)
# print('ars_swimmer_data[-1]=',ars_swimmer_data[-1])
# print('\n',cma_swimmer_data)
# plt.figure()
# plt.title('ES Methods on Swimmer_v2 Environment')
# plt.plot(x,ars_swimmer_data[:200],label='ARS')
# plt.plot(x,vanilla_swimmer_data[:200],label='Vanilla ES')
# plt.plot(x,cdpp_swimmer_data[:200],label='ES with CDPP')
# plt.plot(x,rank_swimmer_data[:200],label='ES with Rank')
# plt.plot(x,cma_swimmer_data[:200],label='CMA-ES')
# plt.legend(loc='lower right')
# plt.xlabel('iteration')
# plt.ylabel('reward')
# plt.show()

# plt.figure()
# plt.title('ES Methods on Swimmer_v2 Environment')
# plt.plot(np.arange(501),ars_swimmer_data,label='ARS')
# plt.plot(x,vanilla_swimmer_data[:200],label='Vanilla ES')
# plt.plot(x,cdpp_swimmer_data[:200],label='ES with CDPP')
# plt.plot(x,rank_swimmer_data[:200],label='ES with Rank')
# plt.plot(x,cma_swimmer_data[:200],label='CMA-ES')
# plt.legend(loc='lower right')
# plt.xlabel('iteration')
# plt.ylabel('reward')
# plt.show()


