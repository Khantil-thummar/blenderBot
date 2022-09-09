import pandas as pd



# Model Training Parameter

batchsize_ = 12
lr_ = 1e-5
optimizer_ = 'adam'
validation_metric_ = 'ppl'
max_train_time_ = 1000
validation_every_n_epochs_ = 0.25


# Data Preprocess Function

def preprocessData(data_path):
    out = []
    path = data_path
    data  = pd.read_csv(path, names=["status", "msg", "time", "id"], header=None)

    valid_conversation = 0
    invalid_conversation = 0

    for user_id_index in data["id"].unique():
        session = [("", "")]
        sent_count = 0
        received_count = 0

        filtered_df = data[data["id"] == user_id_index]
        last_flag = ""
        for i in range(len(filtered_df)):
            status = filtered_df.iloc[i, 0]
            msg = filtered_df.iloc[i, 1]
            if status == "Sent" and str(msg) != "nan":
                sent_count += 1
                if last_flag != status and session[-1][0] != "":
                    if session[-1][1] != "":
                        session[-1][-1] = session[-1][-1] + "\n " + msg
                    else:
                        session[-1][-1] = msg

                last_flag = status

            elif status == "Received" and str(msg) != "nan":
                received_count += 1
                if last_flag != status:
                    session.append([msg, ""])
                else:
                    session[-1][0] = session[-1][0] + "\n " + msg
                last_flag = status

        if sent_count > 0 and received_count > 0:
            valid_conversation += 1
            out.append(session[1:])
        else:
            invalid_conversation += 1

    print(valid_conversation, invalid_conversation)  
    return out  
