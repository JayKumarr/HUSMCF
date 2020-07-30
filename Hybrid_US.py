# @author Jay Kumar

# A hybrid user similarity for collaborative filtering
import statistics as stats
import math


def user_similarity(user_item_rating, sigma = 0.000009):
    user_rating_median = {}  #<userID, Rat_median>
    user_rating_mean = {}
    user_rating_stdev = {}   # <userID, standard_deviation>

    item_user_rating = {}    #[ itemID: <userID  rating1>, ....]
    item_avg_rating = {}    # < itemID, Avg_rating>
    all_ratings = set()
    for u, i in user_item_rating.items():
        for item_key, item_rating in i.items():
            if (item_key not in item_user_rating):
                item_user_rating[item_key] = {}
            item_user_rating[item_key][u] = item_rating
            all_ratings.add(item_rating)

        user_ratings = i.values()
        user_rating_median[u] = stats.median(user_ratings)
        user_rating_mean[u]   = stats.mean(user_ratings)
        try:
            user_rating_stdev[u] = stats.stdev(user_ratings)
        except:
            print("UserID: "+str(u)+" has only one rating in the dataset, therefore it cannot take standard deviation of its ratings")
            exit(0)

    r_med = stats.median(all_ratings)
    #calculating average rating of items
    for i,u in item_user_rating.items():
        item_ratings=u.values()
        item_avg_rating[i]=stats.mean(item_ratings)

    print(all_ratings)
    S3_matrix = calculate_S3_matrix(user_item_rating,user_rating_mean, user_rating_stdev)
    print("S3_matrix")
    S2_matrix = calculate_S2_matrix(user_item_rating)
    print("S2_matrix")
    S_item_matrix = calculate_S_item(item_user_rating, all_ratings, sigma)
    print("Sitem_matrix")

    similarity_u2u_matrix = {}
    for userU, userU_ir in user_item_rating.items():
        similarity_u2u_matrix[userU] = {}
        for userV, userV_jr in user_item_rating.items():
            if (userV == userU):
                continue
            sumParent = 0
            for userU_item, userU_i_rating in userU_ir.items():
                sumChild = 0
                for userV_item, userV_j_rating in userV_jr.items():
                    # PSS implementation
                    s_item_ij = S_item_matrix[userU_item][userV_item]
                    proximity = 1 - (1 /  (1+math.exp(-abs(userU_i_rating-userV_j_rating))) )
                    significance = 1 / ( 1 + math.exp(-( abs(userU_i_rating - r_med)* abs(userV_j_rating-r_med)) )  )
                    avg_rui_rvj = (userU_i_rating + userV_j_rating)/2
                    avg_mean_rui_rvj = (item_avg_rating[userU_item] +item_avg_rating[userV_item])/2
                    singularity = 1 - ( 1/ (1+math.exp(- abs( avg_rui_rvj - avg_mean_rui_rvj ) )) )

                    s1_rui_rvj = proximity* significance * singularity
                    sumChild += (s_item_ij*s1_rui_rvj)
                sumParent+=sumChild

            similarity_u2u_matrix[userU][userV]= S2_matrix[userU][userV] * S3_matrix[userU][userV] * sumParent

    return similarity_u2u_matrix,user_rating_mean, item_avg_rating


def prediction_value(userU, itemID, similarity_u2u_matrix,user_rating_mean,user_item_rating, item_avg_rating, K=40, replace_with_user_mean=True):
    from collections import Counter
    c = Counter(similarity_u2u_matrix[userU])
    nearest_users = c.most_common(K)
    denominator_sum = 0
    nominator_sum = 0
    p_ui = 0

    for userV, sim in nearest_users:
        if itemID in user_item_rating[userV]:  # if this user has given the rating on that particular item
            r_bar_v=user_rating_mean[userV]
            r_vi = user_item_rating[userV][itemID]
            nominator_sum+= (sim * (r_vi - r_bar_v))
            denominator_sum+=abs(sim)


    if (denominator_sum == 0):# if there is no neighbors who rated this item then replace it with user_rating_mean or item_avg_rating
        if (replace_with_user_mean):
            p_ui = user_rating_mean[userU]
        else:
            p_ui = item_avg_rating[itemID]
    else:
        r_u = user_rating_mean[userU]
        p_ui =  r_u + ( nominator_sum / denominator_sum)

    return p_ui


#########################################################
def calculate_S2_matrix(user_item_rating):
    # calculating S2 matrix
    S2_matrix = {}
    for user1, user1_items in user_item_rating.items():
        S2_matrix[user1] = {}
        for user2, user2_items  in user_item_rating.items():
            common_item_set=set(user1_items).intersection(user2_items)
            uiaf=len(common_item_set) / len(user1_items)
            s2_uv = 1 / ( 1+ math.exp(-uiaf) )
            S2_matrix[user1][user2] = s2_uv

    return S2_matrix
#########################################################

def calculate_S3_matrix(user_item_rating, user_rating_mean, user_rating_stdev):
    # calculating S3 matrix
    S3_matrix = {}   # [userID: <user1, s3_value>, ....]
    for user1 in user_item_rating.keys():
        S3_matrix[user1] = {}
        for user2 in user_item_rating.keys():
            meu = abs(user_rating_mean[user1] - user_rating_mean[user2])
            sigma = abs( user_rating_stdev[user1]-user_rating_stdev[user2] )
            temp =math.exp( - ( meu*sigma )   )
            S3_uv = 1 - ( 1 / (1+temp) )
            S3_matrix[user1][user2] = S3_uv
    return S3_matrix


##
def KLD_ij(itemI_urating,itemJ_urating, maximum_rating, all_ratings, smoothing_factor):
    sum = 0
    rating_scale = all_ratings
    for v in rating_scale:

        piv = list(itemI_urating.values()).count(v) / len(itemI_urating)
        pjv = list(itemJ_urating.values()).count(v) / len(itemJ_urating)

        # smoothing factor on probability
        piv_new =  (smoothing_factor + piv) / (1 + ( smoothing_factor* len(rating_scale)))
        pjv_new = (smoothing_factor + pjv) / (1 + ( smoothing_factor* len(rating_scale)))

        ############################

        sum += piv_new * math.log2(piv_new / pjv_new)
    return sum



def calculate_S_item(item_user_rating, all_ratings, sigma):
    S_item_matrix = {}
    for itemI, itemI_urating in item_user_rating.items():
        S_item_matrix[itemI] = {}
        maximum_rating = max(itemI_urating.values())
        for itemJ, itemJ_urating in item_user_rating.items():
            maximum_rating_given_for_item2 = max(itemJ_urating.values())
            if (maximum_rating < maximum_rating_given_for_item2):
                maximum_rating = maximum_rating_given_for_item2
            Dij = KLD_ij(itemI_urating, itemJ_urating,maximum_rating, all_ratings, smoothing_factor=sigma)
            Dji = KLD_ij(itemJ_urating, itemI_urating,maximum_rating, all_ratings, smoothing_factor=sigma)
            Ds_ij = (Dij + Dji) / 2
            s_item_ij = 1 / (1 + Ds_ij)
            S_item_matrix[itemI][itemJ] = s_item_ij

    return S_item_matrix