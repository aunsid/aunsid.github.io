---
layout: post
title: "System Design Basics"
category: others
topic: system-design
date: 2023-01-01
description: "Scalability, reliability, load balancing, databases, caching, and sharding."
---

# Scalability

What is it that we really mean by scalability?

A service is said to be scalable if when we increase the resources in a system, it results in increased performance in a manner proportional to resources added.
Increasing performance in general means serving more units of work, but it can also be to handle larger units of work, such as when datasets grow.

In distributed systems there are other reasons for adding resources to a system; for example to improve the reliability of the offered service. Introducing redundancy is an important first line of defense against failures. An always-on service is said to be scalable if adding resources to facilitate redundancy does not result in a loss of performance.

Source [All Things Distributed](https://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html)

### Horizontal vs Vertical Scaling

Horizontal scaling means that you are adding more servers -- cloning server.

Vertical scaling means that you are adding more power to your existing server -- adding CPU, RAM, storage etc.

![Alt text](https://www.cloudzero.com/hubfs/blog/horizontal-vs-vertical-scaling.webp)


#### Advantages of horizontal scaling

* **Scaling is easier from a hardware perspective** - All horizontal scaling requires you to do is add additional machines to your current pool. It eliminates the need to analyze which system specifications you need to upgrade.
* **Fewer periods of downtime** - Because you're adding a machine, you don't have to switch the old machine off while scaling. If done effectively, there may never be a need for downtime and clients are less likely to be impacted.
* **Increased resilience and fault tolerance** - Relying on a single node for all your data and operations puts you at a high risk of losing it all when it fails. Distributing it among several nodes saves you from losing it all.
* **Increased performance** - If you are using horizontal scaling to manage your network traffic, it allows for more endpoints for connections, considering that the load will be delegated among multiple machines.

#### Disadvantages of horizontal scaling

* **Increased complexity of maintenance and operation** - Multiple servers are harder to maintain than a single server is. Additionally, you will need to add software for load balancing and possibly virtualization. Backing up your machines may also become a little more complex. You will need to ensure that nodes synchronize and communicate effectively.
* **Increased Initial costs** - Adding new servers is far more expensive than upgrading old ones.

#### Advantages of vertical scaling

* **Cost-effective** - Upgrading a pre-existing server costs less than purchasing a new one. Additionally, you are less likely to add new backup and virtualization software when scaling vertically. Maintenance costs may potentially remain the same too.
* **Less complex process communication** - When a single node handles all the layers of your services, it will not have to synchronize and communicate with other machines to work. This may result in faster responses.
* **Less complicated maintenance** - Not only is maintenance cheaper but it is less complex because of the number of nodes you will need to manage.
* **Less need for software changes** - You are less likely to change how the software on a server works or how it is implemented.

#### Disadvantages of vertical scaling

* **Higher possibility for downtime** - Unless you have a backup server that can handle operations and requests, you will need some considerable downtime to upgrade your machine.
* **Single point of failure** - Having all your operations on a single server increases the risk of losing all your data if a hardware or software failure was to occur.
* **Upgrade limitations** - There is a limitation to how much you can upgrade a machine. Every machine has its threshold for RAM, storage, and processing power.

Source [CLOUDZERO](https://www.cloudzero.com/blog/horizontal-vs-vertical-scaling)


# Reliability
It is the probability a system will fail in a given period. A system is said to be reliable, if it delivers its required service even after a component (software/hardware) fails. In any distributed system, any failing component can be replaced by another healthy component. Ensuring the completion of the requested task.

# Availability
It is the time a system remains operational to perform its required function in a specific period. It is a simple measure of the percentage of time that system, service or a machine remains operational under normal circumstances.

#### Reliability vs Availability
If a system is reliable, it is available. A system can reach high availability without being reliable by minimizing repair time and ensuring that spares are always available when needed.


# Load Balancing
It distributes incoming requests to available servers in order to improve responsiveness and availability of applications. Load balancers are effective at:
* _Preventing requests from going to unhealthy servers._
* _Preventing overloading resources._
* _Helps to eliminate a single point of failure._

Load balancers can be placed at 3 places:
* _Between user and web server._
* _Between web server and internal platform layers like application servers._
* _Between internal platform server and database._

Additional benefits:
* **SSL termination** - Decrypt incoming requests and encrypt server responses so backend servers do not have to perform these potentially expensive operations.
* **Session persistence** - Issue cookies and route a specific client's requests to same instance if the web apps do not keep track of sessions.

#### Load Balancing Algorithms

* **Least Connection Method** - Directs traffic to servers with the fewest active connections.
* **Least Response Time** - Directs traffic to servers that have the fewest active connections and the lowest response times.
* **Least Bandwidth Method** - Directs traffic to server that is serving the least amount of traffic measured in Mbps.
* **Round Robin Method** - Cycles through the list of available servers, sending each new request to the next server.
* **Weighted Round Robin** - Same as Round Robin but weights servers by processing capacity.
* **IP Hash** - A hash of the client IP is calculated to redirect the request to a consistent server.


# Application Layer
Separating out the web layer from the application layer allows you to scale and configure both layers independently.

### Microservices
Independently deployable, small, and modular services. Each service runs a unique process and communicates through a well-defined, lightweight mechanism to serve a business goal.


# Database

## Relational Database Management System (RDBMS)
A relational database like SQL is a collection of items organized in tables.

**ACID** is a set of properties of relational database transactions.
* **Atomicity** - Each transaction is all or nothing.
* **Consistency** - Any transaction will bring the database from one valid state to another.
* **Isolation** - Executing transactions concurrently has the same results as if executed serially.
* **Durability** - Once a transaction is committed it will remain so.


### Master-Slave replication
The master serves reads and writes, replicating writes to one or more slaves, which serve only reads.

![Alt text](https://github.com/donnemartin/system-design-primer/blob/master/images/C9ioGtn.png?raw=true)

### Master-Master replication
Both masters serve reads and writes and coordinate with each other on writes.

![Alt text](https://github.com/donnemartin/system-design-primer/blob/master/images/krAHLGg.png?raw=true)

### Federation
Federation splits up the database by function. Instead of a single monolithic database, you have separate databases per domain (forums, users, products), reducing read/write traffic and replication lag per database.


# Caching
Improves page load times and reduces load on servers and databases. A cache is like short-term memory: limited space but faster than the original data source.

### Cache-aside

The application is responsible for reading and writing from storage. The cache does not interact with storage directly.

![Alt text](https://github.com/donnemartin/system-design-primer/blob/master/images/ONjORqk.png?raw=true)

```python
def get_user(self, user_id):
    user = cache.get("user.{0}", user_id)
    if user is None:
        user = db.query("SELECT * FROM users WHERE user_id = {0}", user_id)
        if user is not None:
            key = "user.{0}".format(user_id)
            cache.set(key, json.dumps(user))
    return user
```

### Write-through

The application uses the cache as the main data store. The cache synchronously writes entries to the database.

![Alt text](https://github.com/donnemartin/system-design-primer/blob/master/images/0vBc0hN.png?raw=true)

### Write-behind (write-back)

Asynchronous write to the database after writing to the cache — higher throughput but risk of data loss on cache failure.

![Alt text](https://github.com/donnemartin/system-design-primer/blob/master/images/rgSrvjG.png?raw=true)
