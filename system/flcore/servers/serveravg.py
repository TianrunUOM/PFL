import time
import wandb
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        
        # 记录是否使用wandb
        self.use_wandb = args.use_wandb


    def train(self):
        # 如果使用wandb，创建一个记录全局学习曲线的图表
        if self.use_wandb:
            wandb.define_metric("round")
            wandb.define_metric("test_accuracy", step_metric="round")
            wandb.define_metric("train_loss", step_metric="round")
            wandb.define_metric("time_cost", step_metric="round")
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # 使用wandb记录评估结果
                if self.use_wandb and len(self.rs_test_acc) > 0:
                    # 全局指标记录
                    global_metrics = {
                        "round": i,
                        "test_accuracy": self.rs_test_acc[-1],
                        "train_loss": self.rs_train_loss[-1] if len(self.rs_train_loss) > 0 else 0,
                        "test_auc": self.rs_test_auc[-1] if len(self.rs_test_auc) > 0 else 0,
                        "global/test_accuracy": self.rs_test_acc[-1],  # 另一种命名空间
                        "global/train_loss": self.rs_train_loss[-1] if len(self.rs_train_loss) > 0 else 0,
                        "global/test_auc": self.rs_test_auc[-1] if len(self.rs_test_auc) > 0 else 0
                    }
                    
                    # 添加历史性能轨迹
                    if len(self.rs_test_acc) > 1:
                        test_acc_history = np.array(self.rs_test_acc)
                        global_metrics.update({
                            "global/best_accuracy_so_far": np.max(test_acc_history),
                            "global/accuracy_improvement": self.rs_test_acc[-1] - self.rs_test_acc[0]
                        })
                    
                    wandb.log(global_metrics)
            
            # 记录参与本轮训练的客户端
            if self.use_wandb:
                selected_client_ids = [client.id for client in self.selected_clients]
                wandb.log({
                    "round": i,
                    "selected_clients": wandb.Table(
                        columns=["客户端ID"],
                        data=[[id] for id in selected_client_ids]
                    ),
                    "num_selected_clients": len(selected_client_ids),
                    "selected_clients_ratio": len(selected_client_ids) / self.num_clients
                })

            # 客户端训练循环
            client_losses = []
            
            for client in self.selected_clients:
                client.train()
                
                # 记录每个客户端的训练损失
                if hasattr(client, 'train_loss'):
                    client_losses.append(client.train_loss)
                    
                # 记录详细的客户端训练信息
                if self.use_wandb:
                    client_train_metrics = {
                        "round": i,
                        f"client_{client.id}/train_loss": client.train_loss if hasattr(client, 'train_loss') else 0,
                        f"client_{client.id}/training_samples": client.train_samples,
                        f"client_{client.id}/learning_rate": client.learning_rate,
                        f"clients/train_loss_client_{client.id}": client.train_loss if hasattr(client, 'train_loss') else 0
                    }
                    wandb.log(client_train_metrics)
            
            # 记录所有客户端的平均训练损失
            if self.use_wandb and client_losses:
                wandb.log({
                    "round": i,
                    "clients/avg_train_loss": sum(client_losses) / len(client_losses),
                    "clients/train_loss_std": np.std(client_losses) if len(client_losses) > 1 else 0
                })

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            round_time = self.Budget[-1]
            print('-'*25, 'time cost', '-'*25, round_time)
            
            # 记录每轮训练时间和客户端参与情况
            if self.use_wandb:
                participation_metrics = {
                    "round": i,
                    "time_cost": round_time,
                    "active_clients": len(self.uploaded_ids),
                    "active_clients_ratio": len(self.uploaded_ids) / len(self.selected_clients) if self.selected_clients else 0,
                    "client_participation/selected_count": len(self.selected_clients),
                    "client_participation/active_count": len(self.uploaded_ids),
                    "client_participation/drop_rate": 1 - len(self.uploaded_ids) / len(self.selected_clients) if self.selected_clients else 0
                }
                
                # 记录活跃客户端ID
                if self.uploaded_ids:
                    participation_metrics["active_client_ids"] = wandb.Table(
                        columns=["客户端ID"],
                        data=[[id] for id in self.uploaded_ids]
                    )
                    
                wandb.log(participation_metrics)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                # 如果提前结束，记录最终结果
                if self.use_wandb:
                    wandb.log({
                        "final_test_accuracy": max(self.rs_test_acc),
                        "final_train_loss": min(self.rs_train_loss) if len(self.rs_train_loss) > 0 else 0,
                        "total_rounds": i
                    })
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        
        # 记录最终结果
        if self.use_wandb:
            wandb.log({
                "final_test_accuracy": max(self.rs_test_acc),
                "final_train_loss": min(self.rs_train_loss) if len(self.rs_train_loss) > 0 else 0,
                "avg_time_per_round": sum(self.Budget[1:])/len(self.Budget[1:])
            })

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
            # 记录新客户端评估结果
            if self.use_wandb and len(self.rs_test_acc) > 0:
                wandb.log({
                    "new_clients_test_accuracy": self.rs_test_acc[-1]
                })
