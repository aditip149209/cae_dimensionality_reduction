-- Create the database if it doesn't exist
-- (The docker-compose 'environment' section already does this, but it's good to be safe)
CREATE DATABASE IF NOT EXISTS cae_lol;

USE cae_lol;

CREATE TABLE IF NOT EXISTS `users` (
  `uid` varchar(255) NOT NULL,
  `username` varchar(255) NOT NULL,
  PRIMARY KEY (`uid`)
) 


-- Create the 'user_images' table based on your screenshot
CREATE TABLE IF NOT EXISTS `user_images` (
  `id` int NOT NULL AUTO_INCREMENT,
  `uid` varchar(255) NOT NULL,
  `original_filename` varchar(512) DEFAULT NULL,
  `reconstructed_image_data` mediumtext,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `uid` (`uid`),
  CONSTRAINT `user_images_ibfk_1` FOREIGN KEY (`uid`) REFERENCES `users` (`uid`) ON DELETE CASCADE
) 

