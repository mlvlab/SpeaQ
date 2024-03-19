import os
import sys
import torch
from detectron2.utils import comm
from detectron2.engine import hooks, HookBase
import logging

class PeriodicCheckpointerWithEval(HookBase):
    def __init__(self, eval_period, eval_function, final_test_function, checkpointer, checkpoint_period, max_to_keep=5):
        self.eval = hooks.EvalHook(eval_period, eval_function)
        self.checkpointer = hooks.PeriodicCheckpointer(checkpointer, checkpoint_period, max_to_keep=max_to_keep)
        self.best_ap = 0.0
        self.final_test_func = final_test_function
        best_model_path = checkpointer.save_dir + 'best_model_final.pth.pth'
        if os.path.isfile(best_model_path):
            best_model = torch.load(best_model_path, map_location=torch.device('cpu'))
            try:
                self.best_ap = best_model['SGMeanRecall@100']
            except:
                self.best_ap = best_model['AP50']
            del best_model
            print ("BEST AP: ", self.best_ap)
        else:
            self.best_ap = 0.0

    def before_train(self):
        self.max_iter = self.trainer.max_iter
        self.checkpointer.max_iter = self.trainer.max_iter

    def _do_eval(self):
        results = self.eval._func()
        comm.synchronize()
        return results

    def _do_final_test(self):
        results = self.final_test_func()
        comm.synchronize()
        return results

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self.eval._period > 0 and next_iter % self.eval._period == 0):
            results = self._do_eval()
            if comm.is_main_process():
                try:
                    print (results)
                    dataset = 'VG_val' if 'VG_val' in results.keys() else 'VG_test'
                    if results['SG']['SGMeanRecall@100'] > self.best_ap:
                        self.best_ap = results['SG']['SGMeanRecall@100']
                        additional_state = {"iteration":self.trainer.iter, "SGMeanRecall@100":self.best_ap}
                        self.checkpointer.checkpointer.save(
                        "best_model_final.pth", **additional_state
                        )
                except:
                    current_ap = results['bbox']['AP50']
                    if current_ap > self.best_ap:
                        self.best_ap = current_ap
                        additional_state = {"iteration":self.trainer.iter, "AP50":self.best_ap}
                        self.checkpointer.checkpointer.save(
                        "best_model_final.pth", **additional_state
                        )
        if comm.is_main_process():
            self.checkpointer.step(self.trainer.iter)
        comm.synchronize()

    def after_train(self):
        final_results = self._do_final_test()
        if comm.is_main_process():
            # logger = logging.getLogger("detectron2")
            # logger.info("Evaluation results for final test set in csv format:")
            # logger.info(final_results)
            print(final_results)
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self.eval._func
