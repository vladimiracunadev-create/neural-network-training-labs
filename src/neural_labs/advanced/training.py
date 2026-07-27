from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ..catalog import ROOT
from ..core.protocol import ExperimentLock, SeedPlan, stable_payload_hash
from ..runtime import get_device, save_json, seed_everything
from .catalog import get_track
from .datasets import AdvancedData, load_advanced_data, loader


def _run_dir(track_id: str, output_dir: str | Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = ROOT / output_dir / track_id / stamp
    counter = 1
    while path.exists():
        path = ROOT / output_dir / track_id / f"{stamp}-{counter}"
        counter += 1
    path.mkdir(parents=True)
    return path


def _freeze(track_id: str, run_dir: Path, seeds: SeedPlan, selection_metric: str, data: AdvancedData) -> Path:
    payload_hash = stable_payload_hash({"track": track_id, "metadata": data.metadata, "counts": [len(data.train), len(data.validation), len(data.test)]})
    return ExperimentLock.create(
        lab_id=track_id,
        seeds=seeds,
        config_name="advanced",
        selection_metric=selection_metric,
        selected_checkpoint=run_dir / "best_model.pt",
        dataset_hash=payload_hash,
    ).write(run_dir)


def _classification_eval(model: nn.Module, dataset: Any, device: torch.device, batch_size: int = 64) -> dict[str, float]:
    model.eval(); correct = 0; total = 0; loss_total = 0.0
    with torch.inference_mode():
        for features, targets in loader(dataset, batch_size, False):
            logits = model(features.to(device)); targets = targets.to(device).long()
            loss_total += float(nn.functional.cross_entropy(logits, targets, reduction="sum").item())
            correct += int((logits.argmax(1) == targets).sum().item()); total += len(targets)
    return {"accuracy": correct / max(total, 1), "loss": loss_total / max(total, 1)}


def _train_segmentation(data: AdvancedData, run_dir: Path, device: torch.device, quick: bool, freeze: Callable[[], Path]) -> dict[str, Any]:
    from ..domains.vision.segmentation import UNetSmall, mean_iou

    model = UNetSmall(classes=int(data.classes or 3), base=16 if quick else 32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 1 if quick else 20; best_iou = -math.inf; best_state = copy.deepcopy(model.state_dict()); history = []
    for epoch in range(epochs):
        model.train(); losses=[]
        for images, masks in loader(data.train, 8 if quick else 16, True):
            images,masks=images.to(device),masks.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(images); loss=nn.functional.cross_entropy(logits,masks); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        model.eval(); ious=[]
        with torch.inference_mode():
            for images,masks in loader(data.validation,8,False): ious.append(mean_iou(model(images.to(device)),masks.to(device),int(data.classes or 3)))
        score=float(np.mean(ious)); history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_mean_iou":score})
        if score>best_iou: best_iou=score; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); torch.save({"state_dict":model.state_dict()},run_dir/"best_model.pt"); freeze()
    test_ious=[]; pixel_correct=0; pixels=0
    with torch.inference_mode():
        for images,masks in loader(data.test,8,False):
            logits=model(images.to(device)); targets=masks.to(device); test_ious.append(mean_iou(logits,targets,int(data.classes or 3))); pred=logits.argmax(1); pixel_correct+=int((pred==targets).sum()); pixels+=targets.numel()
    save_json(run_dir/"history.json",history)
    return {"validation_mean_iou":best_iou,"test_mean_iou":float(np.mean(test_ious)),"pixel_accuracy":pixel_correct/max(pixels,1)}


def _train_audio(data: AdvancedData, run_dir: Path, device: torch.device, quick: bool, freeze: Callable[[], Path]) -> dict[str, Any]:
    from ..domains.audio.models import AudioCommandCNN

    model=AudioCommandCNN(int(data.classes or 1)).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    epochs=1 if quick else 15; best=-math.inf; best_state=copy.deepcopy(model.state_dict()); history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for waveform,target in loader(data.train,32,True):
            waveform,target=waveform.to(device),target.to(device).long(); optimizer.zero_grad(set_to_none=True); logits=model(waveform); loss=nn.functional.cross_entropy(logits,target); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        val=_classification_eval(model,data.validation,device,64); history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_accuracy":val["accuracy"]})
        if val["accuracy"]>best: best=val["accuracy"]; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); torch.save({"state_dict":model.state_dict()},run_dir/"best_model.pt"); freeze(); test=_classification_eval(model,data.test,device,64); save_json(run_dir/"history.json",history)
    return {"validation_accuracy":best,"test_accuracy":test["accuracy"],"test_loss":test["loss"]}


def _train_wgan(data: AdvancedData, run_dir: Path, device: torch.device, quick: bool, freeze: Callable[[], Path]) -> dict[str, Any]:
    from ..domains.generative.advanced import WGANGenerator, WGANCritic, gradient_penalty

    latent=64; generator=WGANGenerator(latent).to(device); critic=WGANCritic().to(device); opt_g=torch.optim.Adam(generator.parameters(),lr=1e-4,betas=(0.0,0.9)); opt_c=torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.0,0.9))
    epochs=1 if quick else 20; gp_weight=10.0; history=[]; best=-math.inf; best_state=copy.deepcopy(generator.state_dict())
    for epoch in range(epochs):
        g_losses=[]; c_losses=[]
        for real,_ in loader(data.train,64,True):
            real=real.to(device); batch=len(real)
            for _critic_step in range(2 if quick else 5):
                fake=generator(torch.randn(batch,latent,device=device)).detach(); opt_c.zero_grad(set_to_none=True); c_loss=critic(fake).mean()-critic(real).mean()+gp_weight*gradient_penalty(critic,real,fake); c_loss.backward(); opt_c.step(); c_losses.append(float(c_loss.item()))
            opt_g.zero_grad(set_to_none=True); fake=generator(torch.randn(batch,latent,device=device)); g_loss=-critic(fake).mean(); g_loss.backward(); opt_g.step(); g_losses.append(float(g_loss.item()))
        with torch.inference_mode():
            val_real=next(iter(loader(data.validation,128,False)))[0].to(device); val_fake=generator(torch.randn(len(val_real),latent,device=device)); val_score=float((critic(val_real).mean()-critic(val_fake).mean()).item())
        history.append({"epoch":epoch+1,"generator_loss":float(np.mean(g_losses)),"critic_loss":float(np.mean(c_losses)),"validation_wasserstein":val_score})
        if val_score>best: best=val_score; best_state=copy.deepcopy(generator.state_dict())
    generator.load_state_dict(best_state); torch.save({"generator":generator.state_dict(),"critic":critic.state_dict()},run_dir/"best_model.pt"); freeze()
    with torch.inference_mode():
        real=next(iter(loader(data.test,256,False)))[0].to(device); fake=generator(torch.randn(len(real),latent,device=device)); real_flat=real.flatten(1); fake_flat=fake.flatten(1); mmd=float((torch.cdist(real_flat,real_flat).mean()+torch.cdist(fake_flat,fake_flat).mean()-2*torch.cdist(real_flat,fake_flat).mean()).abs().item())
    save_json(run_dir/"history.json",history); return {"validation_wasserstein":best,"test_energy_distance_proxy":mmd}


def _train_diffusion(data: AdvancedData, run_dir: Path, device: torch.device, quick: bool, freeze: Callable[[], Path]) -> dict[str, Any]:
    from ..domains.generative.advanced import TinyDenoiser, add_noise, cosine_beta_schedule

    steps=100 if quick else 500; model=TinyDenoiser(hidden=32 if quick else 64).to(device); betas=cosine_beta_schedule(steps).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    epochs=1 if quick else 25; best=math.inf; best_state=copy.deepcopy(model.state_dict()); history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for images,_ in loader(data.train,64,True):
            images=images.to(device); t=torch.randint(0,steps,(len(images),),device=device); noise=torch.randn_like(images); noisy=add_noise(images,noise,t,betas); optimizer.zero_grad(set_to_none=True); loss=nn.functional.mse_loss(model(noisy,t),noise); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        model.eval(); val=[]
        with torch.inference_mode():
            for images,_ in loader(data.validation,64,False):
                images=images.to(device); t=torch.randint(0,steps,(len(images),),device=device); noise=torch.randn_like(images); val.append(float(nn.functional.mse_loss(model(add_noise(images,noise,t,betas),t),noise).item()))
        score=float(np.mean(val)); history.append({"epoch":epoch+1,"train_noise_mse":float(np.mean(losses)),"validation_noise_mse":score})
        if score<best: best=score; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); torch.save({"state_dict":model.state_dict(),"betas":betas.cpu()},run_dir/"best_model.pt"); freeze(); test=[]
    with torch.inference_mode():
        for images,_ in loader(data.test,64,False):
            images=images.to(device); t=torch.randint(0,steps,(len(images),),device=device); noise=torch.randn_like(images); test.append(float(nn.functional.mse_loss(model(add_noise(images,noise,t,betas),t),noise).item()))
    save_json(run_dir/"history.json",history); return {"validation_noise_mse":best,"test_noise_mse":float(np.mean(test)),"diffusion_steps":steps}


def _train_simclr(data: AdvancedData, run_dir: Path, device: torch.device, quick: bool, freeze: Callable[[], Path]) -> dict[str, Any]:
    from ..domains.vision.self_supervised import SimCLREncoder, nt_xent_loss

    encoder=SimCLREncoder().to(device); optimizer=torch.optim.AdamW(encoder.parameters(),lr=3e-4); epochs=1 if quick else 50; best=math.inf; best_state=copy.deepcopy(encoder.state_dict()); history=[]
    for epoch in range(epochs):
        encoder.train(); losses=[]
        for first,second,_label in loader(data.train,64,True):
            optimizer.zero_grad(set_to_none=True); _,z1=encoder(first.to(device)); _,z2=encoder(second.to(device)); loss=nt_xent_loss(z1,z2); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        encoder.eval(); val=[]
        with torch.inference_mode():
            for first,second,_label in loader(data.validation,64,False):
                _,z1=encoder(first.to(device)); _,z2=encoder(second.to(device)); val.append(float(nt_xent_loss(z1,z2).item()))
        score=float(np.mean(val)); history.append({"epoch":epoch+1,"train_nt_xent":float(np.mean(losses)),"validation_nt_xent":score})
        if score<best: best=score; best_state=copy.deepcopy(encoder.state_dict())
    encoder.load_state_dict(best_state)
    for parameter in encoder.parameters(): parameter.requires_grad=False
    probe=nn.Linear(128,int(data.classes or 10)).to(device); probe_optimizer=torch.optim.AdamW(probe.parameters(),lr=1e-3)
    for _ in range(1 if quick else 10):
        probe.train()
        for first,_second,label in loader(data.train,64,True):
            with torch.inference_mode(): representation=encoder.encode(first.to(device))
            probe_optimizer.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(probe(representation),label.to(device).long()); loss.backward(); probe_optimizer.step()
    torch.save({"encoder":encoder.state_dict(),"linear_probe":probe.state_dict()},run_dir/"best_model.pt"); freeze()
    correct=0; total=0
    with torch.inference_mode():
        for images,label in loader(data.test,64,False):
            prediction=probe(encoder.encode(images.to(device))).argmax(1); targets=label.to(device); correct+=int((prediction==targets).sum()); total+=len(targets)
    save_json(run_dir/"history.json",history); return {"validation_nt_xent":best,"linear_probe_test_accuracy":correct/max(total,1)}


def _train_transformer(data: AdvancedData, run_dir: Path, device: torch.device, quick: bool, freeze: Callable[[], Path], use_lora: bool) -> dict[str, Any]:
    from ..domains.text.pretrained import build_pretrained_classifier
    try:
        from transformers import DataCollatorWithPadding
    except ImportError as exc:
        raise RuntimeError('Instale el extra text-modern: pip install -e ".[text-modern]"') from exc

    tokenizer=data.metadata["tokenizer"]; collator=DataCollatorWithPadding(tokenizer); model=build_pretrained_classifier("distilbert-base-uncased",int(data.classes or 4),use_lora=use_lora).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    epochs=1 if quick else 3; best=-math.inf; best_state=copy.deepcopy(model.state_dict()); history=[]
    def evaluate_dataset(dataset):
        model.eval(); correct=0; total=0; losses=[]
        with torch.inference_mode():
            for batch in loader(dataset,16,False,collate_fn=collator):
                labels=batch.pop("labels").to(device); inputs={key:value.to(device) for key,value in batch.items()}; output=model(**inputs,labels=labels); losses.append(float(output.loss.item())); correct+=int((output.logits.argmax(1)==labels).sum()); total+=len(labels)
        return correct/max(total,1),float(np.mean(losses))
    for epoch in range(epochs):
        model.train(); losses=[]
        for batch in loader(data.train,16,True,collate_fn=collator):
            labels=batch.pop("labels").to(device); inputs={key:value.to(device) for key,value in batch.items()}; optimizer.zero_grad(set_to_none=True); output=model(**inputs,labels=labels); output.loss.backward(); optimizer.step(); losses.append(float(output.loss.item()))
        accuracy,val_loss=evaluate_dataset(data.validation); history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":val_loss,"validation_accuracy":accuracy})
        if accuracy>best: best=accuracy; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); torch.save({"state_dict":model.state_dict(),"model_name":"distilbert-base-uncased","use_lora":use_lora},run_dir/"best_model.pt"); freeze(); test_accuracy,test_loss=evaluate_dataset(data.test); save_json(run_dir/"history.json",history)
    trainable=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad); total=sum(parameter.numel() for parameter in model.parameters())
    return {"validation_accuracy":best,"test_accuracy":test_accuracy,"test_loss":test_loss,"trainable_parameters":trainable,"total_parameters":total,"use_lora":use_lora}


def train_advanced(
    track_id: str,
    *,
    quick: bool = False,
    split_seed: int = 42,
    training_seed: int = 42,
    device: str = "auto",
    output_dir: str | Path = "runs-advanced",
    use_lora: bool = False,
) -> dict[str, Any]:
    track=get_track(track_id); seeds=SeedPlan(split_seed,training_seed); seed_everything(training_seed); resolved=get_device(device); data=load_advanced_data(track_id,quick=quick,split_seed=split_seed); run_dir=_run_dir(track_id,output_dir)
    save_json(run_dir/"config.json",{"track":track_id,"quick":quick,"split_seed":split_seed,"training_seed":training_seed,"device":str(resolved),"use_lora":use_lora})
    save_json(run_dir/"dataset_manifest.json",{**track,"counts":{"train":len(data.train),"validation":len(data.validation),"test":len(data.test)},"metadata":{key:value for key,value in data.metadata.items() if key!="tokenizer"}})
    selection={"25_transformer_finetuning":"validation_accuracy","26_segmentation_unet":"validation_mean_iou","27_audio_speechcommands":"validation_accuracy","28_wgan_gp":"validation_wasserstein","29_diffusion_ddpm":"validation_noise_mse","30_self_supervised_simclr":"validation_nt_xent"}[track_id]
    freeze=lambda:_freeze(track_id,run_dir,seeds,selection,data)
    if track_id=="25_transformer_finetuning": metrics=_train_transformer(data,run_dir,resolved,quick,freeze,use_lora)
    elif track_id=="26_segmentation_unet": metrics=_train_segmentation(data,run_dir,resolved,quick,freeze)
    elif track_id=="27_audio_speechcommands": metrics=_train_audio(data,run_dir,resolved,quick,freeze)
    elif track_id=="28_wgan_gp": metrics=_train_wgan(data,run_dir,resolved,quick,freeze)
    elif track_id=="29_diffusion_ddpm": metrics=_train_diffusion(data,run_dir,resolved,quick,freeze)
    elif track_id=="30_self_supervised_simclr": metrics=_train_simclr(data,run_dir,resolved,quick,freeze)
    else: raise KeyError(track_id)
    metrics.update({"track":track_id,"device":str(resolved)}); save_json(run_dir/"metrics.json",metrics); return {"run_dir":str(run_dir),"metrics":metrics}
