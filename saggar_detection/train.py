"""
Training script for Saggar Detection Mask R-CNN
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from models import SaggarMaskRCNN


class SaggarTrainer:
    """
    Saggar 감지 모델 학습 클래스
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=0.005,
        num_epochs=50,
        output_dir='./checkpoints'
    ):
        """
        Args:
            model: Mask R-CNN 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            device: 디바이스
            learning_rate: 학습률
            num_epochs: 에포크 수
            output_dir: 체크포인트 저장 디렉토리
        """
        self.model = model.get_model()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Optimizer 설정
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.1
        )

        self.best_loss = float('inf')

    def train_one_epoch(self, epoch):
        """
        한 에포크 학습

        Args:
            epoch (int): 현재 에포크 번호

        Returns:
            float: 평균 손실
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')

        for images, targets in pbar:
            # 데이터를 디바이스로 이동
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()
            num_batches += 1

            # Progress bar 업데이트
            pbar.set_postfix({'loss': losses.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """
        검증

        Returns:
            float: 평균 검증 손실
        """
        self.model.train()  # Validation도 train 모드 (loss 계산을 위해)
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                total_loss += losses.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self):
        """
        전체 학습 프로세스
        """
        print(f"Starting training for {self.num_epochs} epochs...")

        for epoch in range(1, self.num_epochs + 1):
            # 학습
            train_loss = self.train_one_epoch(epoch)
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

            # 검증
            if self.val_loader:
                val_loss = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")

                # Best model 저장
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                    print(f"Best model saved (val_loss: {val_loss:.4f})")

            # Learning rate 업데이트
            self.scheduler.step()

            # 주기적으로 체크포인트 저장
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, train_loss, is_best=False)

        print("Training completed!")

    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        체크포인트 저장

        Args:
            epoch (int): 에포크 번호
            loss (float): 손실 값
            is_best (bool): 최고 성능 모델 여부
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }

        if is_best:
            path = os.path.join(self.output_dir, 'best_model.pth')
        else:
            path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')

        torch.save(checkpoint, path)


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    """
    return tuple(zip(*batch))


def main():
    """
    학습 실행 예제
    """
    # TODO: 실제 데이터셋 로더 구현 필요
    # from dataset import SaggarDataset

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # 데이터셋 로드
    # train_dataset = SaggarDataset('data/train', transforms=get_transform(train=True))
    # val_dataset = SaggarDataset('data/val', transforms=get_transform(train=False))

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=collate_fn
    # )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=2,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=collate_fn
    # )

    # # 모델 초기화
    # model = SaggarMaskRCNN(num_classes=2, device=device)

    # # Trainer 초기화
    # trainer = SaggarTrainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     device=device,
    #     learning_rate=0.005,
    #     num_epochs=50,
    #     output_dir='./checkpoints'
    # )

    # # 학습 시작
    # trainer.train()

    print("Training script template created.")
    print("Please implement the dataset loader for your specific data format.")


if __name__ == '__main__':
    main()
