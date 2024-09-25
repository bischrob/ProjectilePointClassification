# code to sync images using rclone
module load rclone

rclone copy dropbox:Projects/ProjectilePointDatabase/ColoradoProjectilePointdatabase/training_dataset training_dataset --ignore-existing -v

rclone copy dropbox:Projects/ProjectilePointDatabase/ColoradoProjectilePointdatabase/training_dataset_tmp training_dataset_tmp --ignore-existing -v

rclone copy dropbox:Projects/ProjectilePointDatabase/ColoradoProjectilePointdatabase/training_masks training_masks --ignore-existing -v

rclone copy dropbox:Projects/ProjectilePointDatabase/ColoradoProjectilePointdatabase/training_masks_tmp training_masks_tmp --ignore-existing -v

# rclone copy dropbox:Projects/ProjectilePointDatabase/ColoradoProjectilePointdatabase/cropped cropped 