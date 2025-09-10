from custom_predictor.custom_load_inference_source import load_inference_source
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import LOGGER

from pathlib import Path

import cv2
import torch

from ultralytics.data import load_inference_source
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import smart_inference_mode

STREAM_WARNING = """
inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""

class CustomBasePredictor(BasePredictor): 
    # @smart_inference_mode()
    # def stream_inference(self, source=None, model=None, *args, **kwargs):
    #     """
    #     Stream real-time inference on camera feed and save results to file.

    #     Args:
    #         source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
    #             Source for inference.
    #         model (str | Path | torch.nn.Module, optional): Model for inference.
    #         *args (Any): Additional arguments for the inference method.
    #         **kwargs (Any): Additional keyword arguments for the inference method.

    #     Yields:
    #         (ultralytics.engine.results.Results): Results objects.
    #     """
    #     if self.args.verbose:
    #         LOGGER.info("")

    #     # Setup model
    #     if not self.model:
    #         self.setup_model(model)

    #     with self._lock:  # for thread-safe inference
    #         # Setup source every time predict is called
    #         self.setup_source(source if source is not None else self.args.source)

    #         # Check if save_dir/ label file exists
    #         if self.args.save or self.args.save_txt:
    #             (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

    #         # Warmup model
    #         if not self.done_warmup:
    #             self.model.warmup(
    #                 imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, self.model.ch, *self.imgsz)
    #             )
    #             self.done_warmup = True

    #         self.seen, self.windows, self.batch = 0, [], None
    #         profilers = (
    #             ops.Profile(device=self.device),
    #             ops.Profile(device=self.device),
    #             ops.Profile(device=self.device),
    #         )
    #         self.run_callbacks("on_predict_start")
    #         for self.batch in self.dataset:
    #             self.run_callbacks("on_predict_batch_start")
    #             paths, im0s, s = self.batch

    #             # Preprocess
    #             with profilers[0]:
    #                 im = self.preprocess(im0s)

    #             # Inference
    #             with profilers[1]:
    #                 preds = self.inference(im, *args, **kwargs)
    #                 if self.args.embed:
    #                     yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
    #                     continue

    #             # Postprocess
    #             with profilers[2]:
    #                 self.results = self.postprocess(preds, im, im0s)
    #             self.run_callbacks("on_predict_postprocess_end")

    #             # Visualize, save, write results
    #             n = len(im0s)
    #             try:
    #                 for i in range(n):
    #                     self.seen += 1
    #                     self.results[i].speed = {
    #                         "preprocess": profilers[0].dt * 1e3 / n,
    #                         "inference": profilers[1].dt * 1e3 / n,
    #                         "postprocess": profilers[2].dt * 1e3 / n,
    #                     }
    #                     if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
    #                         s[i] += self.write_results(i, Path(paths[i]), im, s)
    #             except StopIteration:
    #                 break

    #             # Print batch results
    #             if self.args.verbose:
    #                 LOGGER.info("\n".join(s))

    #             self.run_callbacks("on_predict_batch_end")
    #             yield from self.results

    #     # Release assets
    #     for v in self.vid_writer.values():
    #         if isinstance(v, cv2.VideoWriter):
    #             v.release()

    #     if self.args.show:
    #         cv2.destroyAllWindows()  # close any open windows

    #     # Print final results
    #     if self.args.verbose and self.seen:
    #         t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
    #         LOGGER.info(
    #             f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
    #             f"{(min(self.args.batch, self.seen), getattr(self.model, 'ch', 3), *im.shape[2:])}" % t
    #         )
    #     if self.args.save or self.args.save_txt or self.args.save_crop:
    #         nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
    #         s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
    #         LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
    #     self.run_callbacks("on_predict_end")

    def setup_source(self, source): 
        """
        Set up source and inference mode.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor):
                Source for inference.
        """

        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
            channels=getattr(self.model, "ch", 4),
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}