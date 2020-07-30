import Hybrid_US as hyb
import recutils as ru
import math

# Example of Paper data
user_item_rating = {}
user_item_rating[1] = { 1:4, 2:3, 3:5, 4:4 }
user_item_rating[2] = { 1:5, 2:3}
user_item_rating[3] = { 1:4, 2:3, 3:3, 4:4}
user_item_rating[4] = { 1:2, 2:1}
user_item_rating[5] = { 1:4, 2:2 , 6:3}
#############################################

CSV_path = 'R1.csv'
# user_item_rating = ru.reading_dataset(CSV_path)




u2u_matrixx,user_rating_mean,item_avg_rating  = hyb.user_similarity(user_item_rating)
# print(u2u_matrixx)

K_value = [2]
file_name = "output"+CSV_path
f_out=open(file_name,'w')
f_out.write("K_Value,UserID,MovieID,AR,PR,MAE,RMSE \n")
print("u2u_matrix done..!")
eval_dict = {}

for k in K_value:
    sumAE = 0
    sumSAE = 0
    count_iteration = 0
    eval_dict[k] = {}
    for u,ir in user_item_rating.items():
        for item, actual_rating in ir.items():
            count_iteration+=1
            pred_rating=hyb.prediction_value(u, item, u2u_matrixx,user_rating_mean,user_item_rating,item_avg_rating, K=k, replace_with_user_mean = True)
            AE = abs(actual_rating-pred_rating)
            sumAE+=AE
            SAE = (AE*AE)
            sumSAE+=SAE
            f_out.write(str(k)+","+str(u)+","+str(item)+","+str(actual_rating)+","+str(pred_rating)+","+str(AE)+","+str(SAE))
            f_out.write("\n")
            # break
            
    MAE = sumAE/count_iteration
    RMSE = sumSAE/count_iteration
    eval_dict[k] = {"MAE":MAE, "RMSE":math.sqrt(RMSE)}


for k, v in eval_dict.items():
    f_out.write(str(k)+", , , , ,"+str(v["MAE"])+","+str(v["RMSE"]))
    f_out.write("\n")

f_out.close()
            
        
    


            
