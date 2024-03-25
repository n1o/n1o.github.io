---
title: "Nixos for Hobby Project"
date: 2023-09-13T13:09:10+02:00
draft: false
tags: ["NixOS"]
---

Lately, I've embarked on a side project: [CodeBreakers](https://codebreakers.re/courses/). It's nothing too fancy, just a website where I plan to release videos and articles about my latest passions—Reverse Engineering and Binary Exploitation.

Creating a website isn't all that complex, and I've done it a couple of times before. The main challenge was deciding where and how to host it. Initially, I considered using [Fly](https://fly.io/) ((which is a fantastic PaaS product with many built-in features), but ultimately, I opted for [Hetzner](hetzner.com) and rented a virtual server. 

Managing a virtual server is definitely a more hands-on solution (and it comes with its fair share of challenges), but it provided me with the opportunity to explore the idea of using [NixOS](https://nixos.org/) on my server.

# Nixos

I used NixOS as my primary OS for around two years, and I must admit, it had its moments of difficulty. For those unfamiliar with it, NixOS is a unique Linux distribution built around the  [Nix](https://nix.dev/tutorials/first-steps/nix-language)programming language. What sets it apart is that your entire operating system configuration is expressed as code.

You might ask:

"Isn't having the operating system as code similar to tools like Ansible, Puppet, or SaltStack?" 

Well, it depends. The technologies mentioned above are more sophisticated and offer a broader range of features. With NixOS, the configuration is more akin to something like Terraform. Essentially, you have a set of features that you can enable and configure (with support for custom modules, so the sky's the limit).


# My Setup 
Now, with a server running NixOS (if you don't have it yet, you can [Infect](https://github.com/elitak/nixos-infect))), here's what I want to achieve:

1. Ngnix
2. Postgresql
3. Django

At Hetzner, I acquired a virtual server for just under 6 Euros, equipped with 2 vCPUs, 4 GB of RAM, and a fixed IP. This is more than sufficient to run everything I need on a single machine. Plus, if I require additional resources, I can efficiently scale vertically. Running all my services on a single machine allows them to communicate through Unix sockets, and the only open ports I need are:


```nix
networking.firewall = {
    enable = true;
    allowedTCPPorts = [22 80 443];
  };
``` 
NixOS, by default, prioritizes security. Consequently, any ports you wish to open must be explicitly defined.


## Ngnix

I use Let's Encrypt certificates to keep costs as low as possible.

```nix
services.nginx = {
    enable = true;
    virtualHosts = {
	    "codebreakers.re" = {
	        addSSL = true;
	        enableACME = true;
	        locations."/.well-known/acme-challenge" = {
	        root = "/var/lib/acme/.challenges";
	        };
	    locations."/static/" = {
	    	alias = "/www/codebreakers/staticfiles/";
            extraConfig  = ''
            expires 1h;
            add_header Cache-Control "public";
            '';
        };
	    locations."/" = {
	    	proxyPass = "http://unix:/run/gunicorn.sock";
	    };
	  };
	};
};

users.users.nginx.extraGroups = [ "acme" ];
security.acme = {
    acceptTerms = true;
    email = "my@email.com";
};
```
Under the hood, there is an ```nginx``` Linux user created, under which Nginx runs as a systemd daemon. Instead of having an ```nginx.conf``` file, our configuration is inlined.

## Postgresql

```nix
services.postgresql = {
  enable = true;
  package = pkgs.postgresql_15;
  ensureDatabases = [ "codebreakers" ];
  authentication = pkgs.lib.mkOverride 10 ''
    local all       all     trust
  '';

  initialScript = pkgs.writeText "backend-initScript" ''
    CREATE USER codebreakers WITH PASSWORD 'password';
    GRANT ALL PRIVILEGES ON DATABASE codebreakers TO codebreakers;
  '';
};
```
Similarly to Nginx, under the hood, a ```postgres``` user is created. By default, this user doesn't open the typical ```5432``` port or allow TCP connections. However, we won't need that, as we can communicate with it through a Unix socket located at ```/var/run/postgresql/.s.PGSQL.5432```.

## Django

This was one of my pain points in the past. While working on Python projects on NixOS (This is especially true if you need libraries like Pytorch), the experience was anything but friendly. Consequently, I eventually decided to run my application within a Docker container (technically using [Podman](https://podman.io/)).

```nix
{

virtualisation.oci-containers.containers = {
   djnago = {
	image = "ghcr.io/n1o/codebreakers-re:$TAG";
	entrypoint = "poetry";
	cmd = [
		"run" 
		"gunicorn" 
		"codebreakers.wsgi" 
		"-w" "2" 
		"-b" "unix:/run/gunicorn.sock"
		"--access-logfile" "'-'" 
		"--error-logfile" "'-' "
		];

	login = {
		username = "n1o";
		passwordFile = "/etc/nixos/github-container-password.txt";

	};
    user = "django";

	volumes = [
		"/var/run/postgresql/:/var/run/postgresql/"
		"/run/:/run/"
	];
   };
};
}
```

This tells Nixos to run ```django``` container as a systemd daemon.


# Release or CI/CD

I store all my code in a single GitHub repository. I have a GitHub action that runs every time I create a new Git tag. This action collects static Django assets, builds my Docker image, which is then pushed to GitHub's Container Repository. It replaces the ```$TAG``` placeholder with the appropriate value and copies the NixOS configuration files to the server through SCP. Finally, on the server, it executes ```nixos-rebuild switch``` as the last step. And the new version is live. And if the switch fails than we stay at the version as before.


# Is NixOS for your
Yes it is! Ok so it has some gotchas but if I compare to tools I used before this is a definite improvement. 

# Disclaimer
I did write this article, but since my writing is far from being perfect I used Chat GPT to improve my writing. 