terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

# Credentials come from the local named profile (see variables.tf); no keys are
# ever written into this configuration.
provider "aws" {
  region  = var.region
  profile = var.profile
}
